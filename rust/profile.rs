use std::collections::HashMap;
use std::fmt;
use std::sync::{Mutex, Once};
use std::time::Instant;

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{subscriber, Subscriber};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;

#[derive(Debug, Clone)]
pub struct ProfileSpan {
    pub name: String,
    pub dur_ns: u64,
    pub device: String,
    pub count: usize,
    pub meta: HashMap<String, String>,
    pub children: Vec<ProfileSpan>,
}

#[derive(Debug)]
struct SpanNode {
    name: String,
    dur_ns: u64,
    start: Option<Instant>,
    device: String,
    count: usize,
    meta: HashMap<String, String>,
    children: Vec<u64>,
}

#[derive(Debug, Default)]
struct CollectorState {
    spans: HashMap<u64, SpanNode>,
    roots: Vec<u64>,
}

static INSTALL: Once = Once::new();
static ACTIVE: Mutex<Option<CollectorState>> = Mutex::new(None);

pub fn begin() {
    install();
    *ACTIVE.lock().expect("profile collector poisoned") = Some(CollectorState::default());
}

pub fn take() -> Vec<ProfileSpan> {
    let mut active = ACTIVE.lock().expect("profile collector poisoned");
    let Some(state) = active.take() else {
        return Vec::new();
    };
    state
        .roots
        .iter()
        .filter_map(|id| build_tree(*id, &state.spans))
        .collect()
}

fn install() {
    INSTALL.call_once(|| {
        let subscriber = tracing_subscriber::registry().with(ProfileLayer);
        let _ = subscriber::set_global_default(subscriber);
    });
}

fn build_tree(id: u64, spans: &HashMap<u64, SpanNode>) -> Option<ProfileSpan> {
    let node = spans.get(&id)?;
    Some(ProfileSpan {
        name: node.name.clone(),
        dur_ns: node.dur_ns,
        device: node.device.clone(),
        count: node.count,
        meta: node.meta.clone(),
        children: node
            .children
            .iter()
            .filter_map(|child| build_tree(*child, spans))
            .collect(),
    })
}

struct ProfileLayer;

impl<S> Layer<S> for ProfileLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut active = ACTIVE.lock().expect("profile collector poisoned");
        let Some(state) = active.as_mut() else {
            return;
        };

        let id_u64 = id.into_u64();
        let mut visitor = FieldVisitor::default();
        attrs.record(&mut visitor);

        let parent_id = attrs
            .parent()
            .map(Id::into_u64)
            .or_else(|| ctx.current_span().id().map(Id::into_u64));

        let node = SpanNode {
            name: attrs.metadata().name().to_string(),
            dur_ns: 0,
            start: None,
            device: visitor
                .meta
                .remove("device")
                .unwrap_or_else(|| "cpu".to_string()),
            count: visitor
                .meta
                .remove("count")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(1),
            meta: visitor.meta,
            children: Vec::new(),
        };
        state.spans.insert(id_u64, node);

        if let Some(parent_id) = parent_id {
            if let Some(parent) = state.spans.get_mut(&parent_id) {
                parent.children.push(id_u64);
                return;
            }
        }
        state.roots.push(id_u64);
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, _ctx: Context<'_, S>) {
        let mut active = ACTIVE.lock().expect("profile collector poisoned");
        let Some(state) = active.as_mut() else {
            return;
        };
        let Some(node) = state.spans.get_mut(&id.into_u64()) else {
            return;
        };
        let mut visitor = FieldVisitor::default();
        values.record(&mut visitor);
        node.meta.extend(visitor.meta);
    }

    fn on_enter(&self, id: &Id, _ctx: Context<'_, S>) {
        let mut active = ACTIVE.lock().expect("profile collector poisoned");
        let Some(state) = active.as_mut() else {
            return;
        };
        if let Some(node) = state.spans.get_mut(&id.into_u64()) {
            node.start = Some(Instant::now());
        }
    }

    fn on_exit(&self, id: &Id, _ctx: Context<'_, S>) {
        let mut active = ACTIVE.lock().expect("profile collector poisoned");
        let Some(state) = active.as_mut() else {
            return;
        };
        if let Some(node) = state.spans.get_mut(&id.into_u64()) {
            if let Some(start) = node.start.take() {
                let ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                node.dur_ns = node.dur_ns.saturating_add(ns);
            }
        }
    }
}

#[derive(Default)]
struct FieldVisitor {
    meta: HashMap<String, String>,
}

impl Visit for FieldVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.meta
            .insert(field.name().to_string(), format!("{value:?}"));
    }
}
