// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration-style replay tests for the native S3 request-trace sink.
//!
//! These exercise the public `S3RequestTraceSink` constructor with a canned
//! AWS HTTP client so CI does not need a live bucket. The same coverage lives
//! as unit tests under `request_trace::sink::tests::s3`; this file exists so
//! the change is visible as a dedicated test target path.

#![cfg(feature = "request-trace-s3")]

use std::io::Read;
use std::time::Duration;

use aws_sdk_s3::config::http::{HttpRequest, HttpResponse};
use aws_sdk_s3::primitives::SdkBody;
use aws_smithy_http_client::test_util::{ReplayEvent, StaticReplayClient};
use dynamo_llm::request_trace::sink::{RequestTraceSink, S3RequestTraceSink};
use dynamo_llm::request_trace::{
    RequestReplayMetrics, RequestTraceEventType, RequestTraceMetrics, RequestTraceRecord,
    RequestTraceSchema,
};
use dynamo_llm::telemetry::jsonl_gz::JsonlGzipSinkOptions;
use dynamo_llm::telemetry::s3_segment_sink::{S3SegmentIdentity, S3SegmentSinkConfig};
use flate2::read::MultiGzDecoder;

const BUCKET: &str = "test-bucket";
const PREFIX: &str = "test-prefix";

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
    // Virtual-hosted style carries ?x-id=PutObject; strip host + query first.
    let path = uri
        .split('?')
        .next()
        .unwrap_or(uri)
        .trim_start_matches("https://")
        .trim_start_matches("http://");
    let object_path = path.split_once('/').map(|(_, rest)| rest).unwrap_or(path);

    let expected_object = format!("pod-a-deadbeef-{seq:06}.jsonl.gz");
    assert!(
        object_path.ends_with(&expected_object),
        "uri {uri:?} object path {object_path:?} should end with {expected_object:?}"
    );
    assert!(
        object_path.contains(&format!("{PREFIX}/")),
        "uri {uri:?} object path {object_path:?} should contain prefix {PREFIX}/"
    );

    let rest = object_path
        .strip_prefix(&format!("{PREFIX}/"))
        .or_else(|| object_path.strip_prefix(&format!("{BUCKET}/{PREFIX}/")))
        .unwrap_or_else(|| panic!("uri {uri:?} missing {PREFIX}/ in {object_path:?}"));
    let parts: Vec<&str> = rest.split('/').collect();
    assert_eq!(
        parts.len(),
        5,
        "expected YYYY/MM/DD/HH/object in {rest:?} (from {uri:?})"
    );
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
    assert!(
        body.contains("\"schema\":\"dynamo.request.trace.v1\""),
        "{body}"
    );
    assert!(body.contains("\"request_id\":\"req-123\""), "{body}");
}

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
