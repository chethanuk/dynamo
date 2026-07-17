// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Write;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use anyhow::{Context as _, anyhow};
use async_nats::jetstream;
use async_trait::async_trait;
use dynamo_runtime::config::environment_names::llm::request_trace as env_request_trace;
use dynamo_runtime::transports::nats;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::telemetry::jsonl::{JsonlSinkOptions, JsonlWriter};
use crate::telemetry::jsonl_gz::{JsonlGzipSinkOptions, JsonlGzipWriter};

use super::{
    RequestTraceFileFormat, RequestTracePolicy, RequestTraceRecord, RequestTraceSinkKind, config,
    otel_sink::OtelRequestTraceSink,
};

static WORKERS_STARTED: AtomicBool = AtomicBool::new(false);

#[async_trait]
pub trait RequestTraceSink: Send + Sync {
    fn name(&self) -> &'static str;
    async fn emit(&self, record: &RequestTraceRecord);
    async fn shutdown(&self) {}
}

pub struct StderrRequestTraceSink;

#[async_trait]
impl RequestTraceSink for StderrRequestTraceSink {
    fn name(&self) -> &'static str {
        "stderr"
    }

    async fn emit(&self, record: &RequestTraceRecord) {
        match serde_json::to_string(record) {
            Ok(json) => {
                if let Err(error) = writeln!(std::io::stderr(), "{json}") {
                    tracing::warn!(%error, "request trace stderr write failed");
                }
            }
            Err(error) => tracing::warn!("request trace serialization failed: {error}"),
        }
    }
}

pub struct NatsRequestTraceSink {
    js: jetstream::Context,
    subject: String,
}

impl NatsRequestTraceSink {
    async fn from_policy(policy: &RequestTracePolicy) -> anyhow::Result<Self> {
        let nats_client = nats::ClientOptions::default()
            .connect()
            .await
            .with_context(|| {
                format!(
                    "Attempting to connect NATS request trace sink from env var {}",
                    env_request_trace::DYN_REQUEST_TRACE_SINKS
                )
            })?;
        Ok(Self {
            js: nats_client.jetstream().clone(),
            subject: policy.nats_subject.clone(),
        })
    }
}

#[async_trait]
impl RequestTraceSink for NatsRequestTraceSink {
    fn name(&self) -> &'static str {
        "nats"
    }

    async fn emit(&self, record: &RequestTraceRecord) {
        match serde_json::to_vec(record) {
            Ok(bytes) => {
                if let Err(error) = self.js.publish(self.subject.clone(), bytes.into()).await {
                    tracing::warn!("request trace nats: publish failed: {error}");
                }
            }
            Err(error) => tracing::warn!("request trace nats: serialize failed: {error}"),
        }
    }
}

pub struct JsonlRequestTraceSink {
    writer: JsonlWriter<RequestTraceRecord>,
}

impl JsonlRequestTraceSink {
    pub async fn new(path: String, options: JsonlSinkOptions) -> anyhow::Result<Self> {
        let writer = JsonlWriter::new(path.clone(), options)
            .await
            .with_context(|| format!("opening jsonl request trace sink at {path}"))?;
        Ok(Self { writer })
    }

    async fn from_policy(policy: &RequestTracePolicy) -> anyhow::Result<Self> {
        let path = policy.file_path.clone().ok_or_else(|| {
            anyhow!(
                "{} must be set when {} includes file",
                env_request_trace::DYN_REQUEST_TRACE_FILE_PATH,
                env_request_trace::DYN_REQUEST_TRACE_SINKS
            )
        })?;
        Self::new(
            path,
            JsonlSinkOptions {
                buffer_bytes: policy.file_buffer_bytes,
                flush_interval: Duration::from_millis(policy.file_flush_interval_ms.max(1)),
            },
        )
        .await
    }
}

#[async_trait]
impl RequestTraceSink for JsonlRequestTraceSink {
    fn name(&self) -> &'static str {
        "file"
    }

    async fn emit(&self, record: &RequestTraceRecord) {
        if self.writer.send(record.clone()).await.is_err() {
            tracing::warn!("request trace file sink closed; dropping record");
        }
    }
}

pub struct JsonlGzipRequestTraceSink {
    writer: JsonlGzipWriter<RequestTraceRecord>,
}

impl JsonlGzipRequestTraceSink {
    pub async fn new(path: String, options: JsonlGzipSinkOptions) -> anyhow::Result<Self> {
        let writer = JsonlGzipWriter::new(path.clone(), options)
            .await
            .with_context(|| format!("opening gzip jsonl request trace sink at {path}"))?;
        Ok(Self { writer })
    }

    async fn from_policy(policy: &RequestTracePolicy) -> anyhow::Result<Self> {
        let path = policy.file_path.clone().ok_or_else(|| {
            anyhow!(
                "{} must be set when {} includes file",
                env_request_trace::DYN_REQUEST_TRACE_FILE_PATH,
                env_request_trace::DYN_REQUEST_TRACE_SINKS
            )
        })?;
        Self::new(
            path,
            JsonlGzipSinkOptions {
                buffer_bytes: policy.file_buffer_bytes,
                flush_interval: Duration::from_millis(policy.file_flush_interval_ms.max(1)),
                roll_uncompressed_bytes: policy.file_roll_bytes,
                roll_lines: policy.file_roll_lines,
                max_segments: None,
            },
        )
        .await
    }
}

#[async_trait]
impl RequestTraceSink for JsonlGzipRequestTraceSink {
    fn name(&self) -> &'static str {
        "file"
    }

    async fn emit(&self, record: &RequestTraceRecord) {
        if self.writer.send(record.clone()).await.is_err() {
            tracing::warn!("request trace file sink closed; dropping record");
        }
    }
}

async fn parse_sinks_from_env() -> anyhow::Result<Vec<Arc<dyn RequestTraceSink>>> {
    let policy = config::policy();
    let mut sinks: Vec<Arc<dyn RequestTraceSink>> = Vec::new();
    for sink_kind in &policy.sinks {
        match sink_kind {
            RequestTraceSinkKind::Stderr => sinks.push(Arc::new(StderrRequestTraceSink)),
            RequestTraceSinkKind::Nats => {
                sinks.push(Arc::new(NatsRequestTraceSink::from_policy(policy).await?))
            }
            RequestTraceSinkKind::Otel => {
                sinks.push(Arc::new(OtelRequestTraceSink::from_policy(policy).await?))
            }
            RequestTraceSinkKind::File => match policy.file_format {
                RequestTraceFileFormat::Jsonl => {
                    sinks.push(Arc::new(JsonlRequestTraceSink::from_policy(policy).await?))
                }
                RequestTraceFileFormat::JsonlGz => sinks.push(Arc::new(
                    JsonlGzipRequestTraceSink::from_policy(policy).await?,
                )),
            },
        }
    }
    Ok(sinks)
}

pub async fn spawn_workers_from_env(shutdown: CancellationToken) -> anyhow::Result<()> {
    if WORKERS_STARTED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Ok(());
    }

    if let Err(error) = spawn_workers(shutdown).await {
        WORKERS_STARTED.store(false, Ordering::Release);
        return Err(error);
    }
    Ok(())
}

async fn spawn_workers(shutdown: CancellationToken) -> anyhow::Result<()> {
    let sinks = parse_sinks_from_env().await?;
    let sink_count = sinks.len();
    for sink in sinks {
        let name = sink.name();
        let mut receiver: broadcast::Receiver<RequestTraceRecord> = super::subscribe();
        let worker_shutdown = shutdown.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    _ = worker_shutdown.cancelled() => {
                        loop {
                            match receiver.try_recv() {
                                Ok(record) => sink.emit(&record).await,
                                Err(broadcast::error::TryRecvError::Lagged(count)) => tracing::warn!(
                                    sink = name,
                                    dropped = count,
                                    "request trace bus lagged during shutdown; dropped records"
                                ),
                                Err(
                                    broadcast::error::TryRecvError::Empty
                                    | broadcast::error::TryRecvError::Closed
                                ) => break,
                            }
                        }
                        break;
                    }
                    message = receiver.recv() => {
                        match message {
                            Ok(record) => sink.emit(&record).await,
                            Err(broadcast::error::RecvError::Lagged(count)) => tracing::warn!(
                                sink = name,
                                dropped = count,
                                "request trace bus lagged; dropped records"
                            ),
                            Err(broadcast::error::RecvError::Closed) => break,
                        }
                    }
                }
            }
            sink.shutdown().await;
        });
    }

    if sink_count == 0 {
        tracing::warn!("request trace is enabled but no valid request trace sinks were configured");
    }
    tracing::info!(sinks = sink_count, "Request trace sinks ready");
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use flate2::read::MultiGzDecoder;
    use tempfile::tempdir;

    use crate::request_trace::RequestReplayMetrics;
    use crate::telemetry::jsonl_gz::segment_path;

    use super::*;
    use crate::request_trace::RequestTraceEventType;
    use crate::request_trace::RequestTraceMetrics;
    use crate::request_trace::RequestTraceSchema;

    fn sample_record() -> RequestTraceRecord {
        RequestTraceRecord {
            schema: RequestTraceSchema::V1,
            event_type: RequestTraceEventType::RequestEnd,
            event_time_unix_ms: 1_100,
            event_source: None,
            agent_context: None,
            request: Some(RequestTraceMetrics {
                request_id: "req-123".to_string(),
                x_request_id: None,
                model: None,
                input_tokens: None,
                output_tokens: Some(7),
                cached_tokens: None,
                request_received_ms: Some(1_000),
                prefill_wait_time_ms: None,
                prefill_time_ms: None,
                ttft_ms: None,
                total_time_ms: None,
                avg_itl_ms: None,
                kv_hit_rate: None,
                kv_transfer_estimated_latency_ms: None,
                queue_depth: None,
                worker: None,
                replay: Some(RequestReplayMetrics {
                    trace_block_size: 2,
                    input_length: 3,
                    input_sequence_hashes: vec![11, 22],
                }),
                finish_reason_metadata: None,
            }),
            tool: None,
            payload: None,
        }
    }

    #[tokio::test]
    async fn jsonl_sink_writes_request_record() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("request_trace.jsonl");
        let sink = JsonlRequestTraceSink::new(
            path.display().to_string(),
            JsonlSinkOptions {
                buffer_bytes: 128,
                flush_interval: Duration::from_millis(10),
            },
        )
        .await
        .unwrap();

        sink.emit(&sample_record()).await;

        let mut content = String::new();
        for _ in 0..100 {
            content = tokio::fs::read_to_string(&path).await.unwrap_or_default();
            if content.contains("\"request_id\":\"req-123\"") {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(content.contains("\"schema\":\"dynamo.request.trace.v1\""));
        assert!(!content.contains("agent_context"));
        assert!(!content.contains("\"tool\""));
    }

    #[tokio::test]
    async fn gzip_sink_writes_and_rolls_request_records() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("request_trace");
        let sink = JsonlGzipRequestTraceSink::new(
            path.display().to_string(),
            JsonlGzipSinkOptions {
                buffer_bytes: 1,
                flush_interval: Duration::from_secs(60),
                roll_uncompressed_bytes: 1024 * 1024,
                roll_lines: Some(1),
                max_segments: None,
            },
        )
        .await
        .unwrap();

        sink.emit(&sample_record()).await;
        sink.emit(&sample_record()).await;

        for index in 0..2 {
            let segment = segment_path(&path, index);
            let mut content = String::new();
            for _ in 0..100 {
                if segment.exists() {
                    let bytes = std::fs::read(&segment).unwrap();
                    let mut decoder = MultiGzDecoder::new(bytes.as_slice());
                    decoder.read_to_string(&mut content).unwrap();
                    if content.contains("\"request_id\":\"req-123\"") {
                        break;
                    }
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            assert!(content.contains("\"request_id\":\"req-123\""));
        }
    }

    #[cfg(feature = "request-trace-s3")]
    mod s3 {
        use aws_sdk_s3::config::http::{HttpRequest, HttpResponse};
        use aws_sdk_s3::primitives::SdkBody;
        use aws_smithy_http_client::test_util::{ReplayEvent, StaticReplayClient};

        use crate::telemetry::s3_segment_sink::{S3SegmentIdentity, S3SegmentSinkConfig};

        use super::*;
        use crate::request_trace::sink::S3RequestTraceSink;

        const BUCKET: &str = "test-bucket";
        const PREFIX: &str = "test-prefix";

        /// One canned `200 OK` for each PUT the sink is expected to make.
        fn replay_client(puts: usize) -> StaticReplayClient {
            StaticReplayClient::new(
                (0..puts)
                    .map(|_| {
                        ReplayEvent::new(
                            HttpRequest::new(SdkBody::empty()),
                            HttpResponse::new(200_u16.try_into().unwrap(), SdkBody::empty()),
                        )
                    })
                    .collect(),
            )
        }

        fn s3_client(replay: &StaticReplayClient) -> aws_sdk_s3::Client {
            let config = aws_sdk_s3::Config::builder()
                .behavior_version(aws_sdk_s3::config::BehaviorVersion::latest())
                .credentials_provider(aws_sdk_s3::config::Credentials::new(
                    "test", "test", None, None, "test",
                ))
                .region(aws_sdk_s3::config::Region::new("us-east-1"))
                .http_client(replay.clone())
                .build();
            aws_sdk_s3::Client::from_conf(config)
        }

        fn test_config() -> S3SegmentSinkConfig {
            S3SegmentSinkConfig {
                bucket: BUCKET.to_string(),
                prefix: PREFIX.to_string(),
                identity: S3SegmentIdentity {
                    instance: "pod-a".to_string(),
                    startup: "deadbeef".to_string(),
                },
            }
        }

        /// Thresholds high enough that nothing but an explicit shutdown can
        /// trigger an upload.
        fn no_roll_options() -> JsonlGzipSinkOptions {
            JsonlGzipSinkOptions {
                buffer_bytes: 1,
                flush_interval: Duration::from_secs(60),
                roll_uncompressed_bytes: 1024 * 1024 * 1024,
                roll_lines: None,
                max_segments: None,
            }
        }

        fn gunzip(bytes: &[u8]) -> String {
            let mut out = String::new();
            MultiGzDecoder::new(bytes).read_to_string(&mut out).unwrap();
            out
        }

        fn assert_key_matches(uri: &str, seq: u64) {
            // {prefix}/YYYY/MM/DD/HH/{instance}-{startup}-{seq:06}.jsonl.gz
            let expected_suffix = format!("/pod-a-deadbeef-{seq:06}.jsonl.gz");
            assert!(
                uri.ends_with(&expected_suffix),
                "uri {uri:?} should end with {expected_suffix:?}"
            );
            let path = uri.split('?').next().unwrap();
            let rest = path
                .strip_prefix(&format!("/{BUCKET}/{PREFIX}/"))
                .unwrap_or_else(|| panic!("uri {uri:?} missing /{BUCKET}/{PREFIX}/ prefix"));
            let parts: Vec<&str> = rest.split('/').collect();
            assert_eq!(parts.len(), 5, "expected YYYY/MM/DD/HH/object in {rest:?}");
            assert!(
                parts[0].len() == 4 && parts[0].chars().all(|c| c.is_ascii_digit()),
                "year {:?} in {rest:?}",
                parts[0]
            );
            for part in &parts[1..4] {
                assert!(
                    part.len() == 2 && part.chars().all(|c| c.is_ascii_digit()),
                    "date part {part:?} in {rest:?}"
                );
            }
        }

        /// The bug: records buffered below the roll threshold are only uploaded
        /// if `shutdown()` flushes and closes the segment. Without that, the
        /// final object never lands.
        #[tokio::test]
        async fn s3_sink_uploads_pending_records_on_shutdown() {
            let replay = replay_client(1);
            let sink = S3RequestTraceSink::new(s3_client(&replay), test_config(), no_roll_options())
                .await
                .unwrap();

            sink.emit(&sample_record()).await;
            sink.shutdown().await;

            let requests: Vec<_> = replay.actual_requests().collect();
            assert_eq!(
                requests.len(),
                1,
                "expected exactly one PUT for the pending segment"
            );
            assert_eq!(requests[0].method(), "PUT");
            assert_key_matches(requests[0].uri(), 0);

            let body = gunzip(requests[0].body().bytes().expect("in-memory body"));
            assert!(body.contains("\"schema\":\"dynamo.request.trace.v1\""), "{body}");
            assert!(body.contains("\"request_id\":\"req-123\""), "{body}");
        }

        /// S3 objects cannot be appended to after the fact, so every roll must
        /// finalize the segment it leaves behind, not just the last one.
        #[tokio::test]
        async fn s3_sink_uploads_each_segment_on_roll() {
            let replay = replay_client(2);
            let options = JsonlGzipSinkOptions {
                roll_lines: Some(1),
                ..no_roll_options()
            };
            let sink = S3RequestTraceSink::new(s3_client(&replay), test_config(), options)
                .await
                .unwrap();

            sink.emit(&sample_record()).await;
            sink.emit(&sample_record()).await;
            sink.shutdown().await;

            let requests: Vec<_> = replay.actual_requests().collect();
            assert_eq!(requests.len(), 2, "expected one PUT per rolled segment");
            for (seq, request) in requests.iter().enumerate() {
                assert_eq!(request.method(), "PUT");
                assert_key_matches(request.uri(), seq as u64);
                let body = gunzip(request.body().bytes().expect("in-memory body"));
                assert!(body.contains("\"request_id\":\"req-123\""), "{body}");
            }
        }

        /// Selecting the s3 sink without a bucket must fail loudly, naming the
        /// variable the operator forgot.
        #[tokio::test]
        async fn s3_sink_without_bucket_fails() {
            let mut policy = config::policy().clone();
            policy.s3_bucket = None;

            let error = S3RequestTraceSink::from_policy(&policy)
                .await
                .expect_err("s3 sink must not build without a bucket");
            assert!(
                error
                    .to_string()
                    .contains(env_request_trace::DYN_REQUEST_TRACE_S3_BUCKET),
                "error should name the missing variable, got: {error}"
            );
        }
    }
}
