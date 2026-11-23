# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Convert a Simpler L2 Chrome trace to native Perfetto TrackEvents."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def _collect_tracks(events):
    processes, labels, ranks = {}, {}, {}
    streams, counters = defaultdict(list), defaultdict(list)
    bars = defaultdict(list)
    by_id = {}
    for event in events:
        pid, phase = event["pid"], event["ph"]
        key = (pid, event.get("tid"), None)
        if phase == "M":
            name, args = event["name"], event["args"]
            if name == "process_name":
                processes[pid] = args["name"]
            elif name == "thread_name":
                labels[key] = args["name"]
            elif name == "thread_sort_index":
                ranks[key] = args["sort_index"]
            continue
        ts = round(event["ts"] * 1000)
        if phase in ("n", "b", "e") and "id2" in event:
            key = (pid, None, event["id2"]["local"])
            if phase == "n":
                labels[key] = event["name"]
            streams[key].append((ts, phase, event))
        elif phase == "X":
            streams[key].append((ts, "b", event))
            streams[key].append((ts + round(event["dur"] * 1000), "e", event))
            bars[key].append(event)
            if "id" in event:
                by_id[key, event["id"]] = event
        elif phase == "C":
            # Overhead rows can be the only events on their process and have no thread metadata.
            processes.setdefault(pid, f"Process {pid}")
            for series, value in event["args"].items():
                counter = (pid, event.get("tid"), event.get("id"), event.get("cat", ""), event["name"], series)
                counters[counter].append((ts, value))
        elif phase == "i":
            streams[key].append((ts, phase, event))
        elif phase not in ("s", "f"):
            raise ValueError(f"Unsupported L2 trace phase {phase!r}")

    # The Chrome importer orders async tracks by name. Native tracks have an
    # explicit sibling rank, so display the member name first within each MIX.
    slots = sorted((key for key in labels if key[2] is not None), key=lambda key: labels[key].lower())
    for rank, key in enumerate(slots):
        ranks[key] = rank
        labels[key] = re.sub(
            r"/ (MIX\d+)(?: \((AIV0 proxy)\)|/(AIC|AIV0|AIV1))$",
            lambda match: f"/ {match[2] or match[3]}/{match[1]}",
            labels[key],
        )
    return processes, labels, ranks, streams, counters, bars, by_id


def _flow_bindings(events, bars, by_id):
    starts, ends = defaultdict(list), defaultdict(list)
    flow_ids = {}
    for event in events:
        if event["ph"] not in ("s", "f"):
            continue
        key = (event["pid"], event["tid"], None)
        anchor = by_id.get((key, event.get("bind_id")))
        if anchor is None and "bind_id" not in event:
            anchor = next((bar for bar in bars[key] if bar["ts"] <= event["ts"] <= bar["ts"] + bar["dur"]), None)
        if anchor is None:
            raise ValueError(f"Missing L2 flow endpoint for {event['id']}")
        bindings = starts if event["ph"] == "s" else ends
        bindings[id(anchor)].append(flow_ids.setdefault(event["id"], len(flow_ids) + 1))
    return starts, ends


def _annotation(fields, name, value):
    field = fields.add(name=name)
    if isinstance(value, bool):
        field.bool_value = value
    elif isinstance(value, int):
        if value < 0:
            field.int_value = value
        else:
            field.uint_value = value
    elif isinstance(value, float):
        field.double_value = value
    elif isinstance(value, str):
        field.string_value = value
    elif isinstance(value, dict):
        for key, item in value.items():
            _annotation(field.dict_entries, key, item)
    elif value is not None:
        field.legacy_json_value = json.dumps(value)


def encode_l2_trace(events):
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace, TrackDescriptor, TrackEvent  # noqa: PLC0415

    trace = Trace()
    processes, labels, ranks, streams, counters, bars, by_id = _collect_tracks(events)
    starts, ends = _flow_bindings(events, bars, by_id)
    parents = {}
    for uuid, (pid, name) in enumerate(sorted(processes.items()), 1):
        desc = trace.packet.add(trusted_packet_sequence_id=1).track_descriptor
        desc.uuid, desc.name = uuid, name
        desc.child_ordering = TrackDescriptor.EXPLICIT
        parents[pid] = uuid

    for uuid, (key, stream) in enumerate(streams.items(), len(parents) + 1):
        desc = trace.packet.add(trusted_packet_sequence_id=1).track_descriptor
        desc.uuid, desc.parent_uuid, desc.name = uuid, parents[key[0]], labels[key]
        if key in ranks:
            desc.sibling_order_rank = ranks[key]
        for ts, phase, event in sorted(stream, key=lambda row: row[0]):
            packet = trace.packet.add(timestamp=ts, trusted_packet_sequence_id=1)
            record = packet.track_event
            record.track_uuid = uuid
            if phase == "e":
                record.type = TrackEvent.TYPE_SLICE_END
                continue
            record.type = TrackEvent.TYPE_SLICE_BEGIN if phase == "b" else TrackEvent.TYPE_INSTANT
            record.name = labels[key] if phase == "n" else event["name"]
            record.categories.append(event["cat"])
            args = event.get("args", {})
            if "correlation_id" in args:
                record.correlation_id_str = args["correlation_id"]
            for name, value in args.items():
                _annotation(record.debug_annotations, name, value)
            record.flow_ids.extend(starts.get(id(event), ()))
            record.terminating_flow_ids.extend(ends.get(id(event), ()))
    for uuid, ((pid, tid, _, category, name, series), samples) in enumerate(
        counters.items(), len(parents) + len(streams) + 1
    ):
        desc = trace.packet.add(trusted_packet_sequence_id=1).track_descriptor
        desc.uuid, desc.parent_uuid = uuid, parents[pid]
        desc.name = name if name == series else f"{name}.{series}"
        desc.counter.is_incremental = False
        if category:
            desc.counter.categories.append(category)
        if (pid, tid, None) in ranks:
            desc.sibling_order_rank = ranks[pid, tid, None]
        for ts, value in sorted(samples, key=lambda sample: sample[0]):
            record = trace.packet.add(timestamp=ts, trusted_packet_sequence_id=1).track_event
            record.type, record.track_uuid = TrackEvent.TYPE_COUNTER, uuid
            if type(value) is int:
                record.counter_value = value
            elif type(value) is float:
                record.double_counter_value = value
            else:
                raise ValueError(f"Invalid L2 counter value for {name}.{series}: {value!r}")
    return trace.SerializeToString()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_bytes(encode_l2_trace(json.loads(args.input.read_bytes())["traceEvents"]))


if __name__ == "__main__":
    main()
