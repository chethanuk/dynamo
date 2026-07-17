// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! S3 implementation of [`SegmentSink`](super::jsonl_gz::SegmentSink).
//!
//! An S3 object cannot be appended to once written, so each segment is
//! accumulated in memory across one or more `append_to_segment` calls and
//! uploaded as a single object on `close_segment`. The body is a concatenation
//! of self-contained gzip members, so it downloads and `gunzip`s as an ordinary
//! `.jsonl.gz` file.
//!
//! Object keys:
//! ```text
//! {prefix}/YYYY/MM/DD/HH/{instance}-{startup}-{seq:06}.jsonl.gz
//! ```
//! `instance` identifies the writing process and `startup` is randomized once
//! per process, so replicas -- and a restarted pod that keeps its name, as a
//! StatefulSet does -- cannot overwrite each other's segments.
//!
//! Failure policy: rely on the SDK's default retry (3 attempts, exponential
//! backoff with jitter). On terminal failure log and drop the segment; never
//! propagate, so one bad upload cannot kill the writer task or the frontend.

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{Context as _, Result};
use async_trait::async_trait;
use aws_sdk_s3::Client;
use aws_sdk_s3::primitives::ByteStream;
use chrono::{DateTime, Datelike, Timelike, Utc};

use super::jsonl_gz::SegmentSink;

/// Identity strings baked into object keys to keep writers from colliding.
#[derive(Clone, Debug)]
pub struct S3SegmentIdentity {
    pub instance: String,
    pub startup: String,
}

impl S3SegmentIdentity {
    /// Resolve the instance id from the Kubernetes downward API, falling back
    /// to the hostname. `startup` is random per process.
    pub fn resolve() -> Self {
        let instance = std::env::var("POD_NAME")
            .ok()
            .or_else(|| std::env::var("HOSTNAME").ok())
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .or_else(|| {
                std::fs::read_to_string("/etc/hostname")
                    .ok()
                    .map(|value| value.trim().to_string())
                    .filter(|value| !value.is_empty())
            })
            .unwrap_or_else(|| "unknown".to_string());

        let startup = uuid::Uuid::new_v4().simple().to_string()[..8].to_string();

        Self { instance, startup }
    }
}

#[derive(Clone, Debug)]
pub struct S3SegmentSinkConfig {
    pub bucket: String,
    pub prefix: String,
    pub identity: S3SegmentIdentity,
}

pub struct S3SegmentSink {
    client: Client,
    config: S3SegmentSinkConfig,
    /// Per-seq accumulator. `append_to_segment` extends the entry; the matching
    /// `close_segment` removes it and uploads.
    segments: Mutex<HashMap<u64, Vec<u8>>>,
}

impl S3SegmentSink {
    pub fn new(client: Client, config: S3SegmentSinkConfig) -> Self {
        Self {
            client,
            config,
            segments: Mutex::new(HashMap::new()),
        }
    }

    /// Build a client from the standard AWS provider chain. `region` overrides
    /// `AWS_REGION`; the endpoint (`AWS_ENDPOINT_URL`) and retry settings
    /// (`AWS_MAX_ATTEMPTS`) are read from the environment by the SDK itself.
    pub async fn build_client(region: Option<String>) -> Client {
        let mut loader = aws_config::defaults(aws_config::BehaviorVersion::latest());
        if let Some(region) = region {
            loader = loader.region(aws_sdk_s3::config::Region::new(region));
        }
        Client::new(&loader.load().await)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<u64, Vec<u8>>> {
        self.segments.lock().unwrap_or_else(|e| e.into_inner())
    }

    async fn upload(&self, key: &str, body: Vec<u8>) -> Result<()> {
        let len = body.len();
        self.client
            .put_object()
            .bucket(&self.config.bucket)
            .key(key)
            .body(ByteStream::from(body))
            // Not Content-Encoding: gzip -- that makes some clients silently
            // decompress, and the .gz suffix already says what this is.
            .content_type("application/gzip")
            .send()
            .await
            .with_context(|| format!("request trace s3 put_object key={key} size={len}"))?;
        Ok(())
    }
}

#[async_trait]
impl SegmentSink for S3SegmentSink {
    async fn append_to_segment(&self, seq: u64, gz_bytes: Vec<u8>) -> Result<()> {
        self.lock()
            .entry(seq)
            .or_default()
            .extend_from_slice(&gz_bytes);
        Ok(())
    }

    async fn close_segment(&self, seq: u64) -> Result<()> {
        // A segment that never took a member (shutdown before any record, or a
        // roll on an exact boundary) has nothing to ship.
        let Some(body) = self.lock().remove(&seq) else {
            return Ok(());
        };
        if body.is_empty() {
            return Ok(());
        }

        let len = body.len();
        let key = format_object_key(
            &self.config.prefix,
            Utc::now(),
            &self.config.identity.instance,
            &self.config.identity.startup,
            seq,
        );

        match self.upload(&key, body).await {
            Ok(()) => tracing::debug!(key, bytes = len, "request trace s3 segment uploaded"),
            // Dropping the segment is the same bargain every other sink makes:
            // tracing must never take the serving path down with it.
            Err(error) => tracing::warn!(
                key,
                bytes = len,
                %error,
                "request trace s3 upload failed; dropping segment"
            ),
        }
        Ok(())
    }
}

fn format_object_key(
    prefix: &str,
    now: DateTime<Utc>,
    instance: &str,
    startup: &str,
    seq: u64,
) -> String {
    let prefix = prefix.trim_matches('/');
    format!(
        "{prefix}/{year:04}/{month:02}/{day:02}/{hour:02}/{instance}-{startup}-{seq:06}.jsonl.gz",
        year = now.year(),
        month = now.month(),
        day = now.day(),
        hour = now.hour(),
    )
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    use super::*;

    #[test]
    fn object_key_is_hour_partitioned_and_writer_scoped() {
        let now = Utc.with_ymd_and_hms(2026, 7, 16, 9, 30, 0).unwrap();
        assert_eq!(
            format_object_key("traces", now, "pod-a", "deadbeef", 42),
            "traces/2026/07/16/09/pod-a-deadbeef-000042.jsonl.gz"
        );
    }

    #[test]
    fn object_key_does_not_double_slash_a_padded_prefix() {
        let now = Utc.with_ymd_and_hms(2026, 1, 2, 3, 0, 0).unwrap();
        assert_eq!(
            format_object_key("/traces/dev/", now, "pod-a", "deadbeef", 0),
            "traces/dev/2026/01/02/03/pod-a-deadbeef-000000.jsonl.gz"
        );
    }
}
