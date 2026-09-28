"""
nmea_sky.py

GSA and GSV: what the receiver can SEE, as opposed to where it thinks it is.

src/gps.py answers "where am I". This module answers the question that matters
when gps.py can answer nothing at all. Indoors - which is where a car in a
garage lives - RMC is void and GGA reports zero satellites used, and those two
facts are byte-identical between a healthy receiver under a concrete deck, a
receiver whose antenna connector has vibrated loose, and a receiver that was
powered up ninety seconds ago and has no almanac yet. GSV separates them:
"in view" is computed from the stored almanac and needs no reception at all,
so satellites in view with every SNR field blank is an antenna or LNA fault,
while satellites in view at SNR 20 with no fix is a sky-view problem and
nothing is broken.

This is a separate module from gps.py deliberately, and it imports nothing from
it. The position parser and the sky parser can then fail independently, and be
tested independently - which matters because the field-index bug and the
accumulation bug below are different bugs and a combined test cannot tell them
apart.

THE FIELD-INDEX CONVENTION
--------------------------
Callers pass fields ALREADY SLICED the way gps.py slices them:

    fields = sentence.strip().split("*")[0].split(",")[1:]

The trailing [1:] has already dropped the "$GPGSV" token, so fields[n] here is
raw NMEA field n+1. Cross-check: gps.py reads GGA numSV from fields[6] and raw
GGA numSV is field 7.

Getting this wrong does not raise. Using the raw base 4+4i on already-sliced
fields shifts every satellite one field left, so SNR reads the AZIMUTH -
which is 000-359 and almost never zero. Every satellite then appears to have
strong signal, and a receiver with a disconnected antenna is reported as
"antenna fine, just needs sky": no exception, no error, exactly the inverted
diagnosis, on the one page that exists to answer that question.
"""
from dataclasses import dataclass, field

# A constellation that stops sending GSV must disappear from the page rather
# than sit there at its last-known SNR forever. Ten seconds is several GSV
# cycles at the usual 1Hz, so it never trips on a single dropped sentence, and
# it is short enough that walking from a window back indoors is visible.
PUBLISHED_CYCLE_EXPIRY_SECONDS = 10.0

# The sentences of one GSV cycle arrive together: the receiver is read over
# CDC-ACM, so a whole burst lands in a single read and its sentences are parsed
# milliseconds apart. A continuation that turns up a second or more after the
# sentence it continues therefore belongs to a DIFFERENT cycle whose earlier
# sentences were lost to a checksum failure - which gps.py drops silently, so a
# lost sentence is an ordinary event and not an emergency. Two seconds is
# longer than any real gap inside a burst and far shorter than the published
# window, which is exactly the point: unbounded, an hour-old partial gets
# extended by sentence 3 of a fresh cycle and republished with a new timestamp,
# and the page renders satellites last heard 80 minutes ago as a live sky.
PARTIAL_CYCLE_MAX_AGE_SECONDS = 2.0

# The GSA epoch ends at the next RMC - but only if that RMC arrives. Lost to a
# checksum failure it never does, and the following burst then merges into the
# epoch still standing open. fix_type is a MAX across the epoch, so a 3D fix
# from one second ago outranks the "no fix" the receiver is reporting now and
# sits on the page beside a void RMC. Half a second falls between two bursts
# and never through one: a burst arrives in one read, while the gap to the next
# burst is a whole fix interval even at 2Hz.
GSA_EPOCH_MAX_AGE_SECONDS = 0.5

# Twelve PRN slots per GSA and at most a handful of systems per epoch, so a
# real epoch never comes near this. It bounds the pathological one: a receiver
# configured to emit no RMC at all holds a single epoch open forever, and a
# pre-4.10 burst tags its sentences by arrival order, so every sentence adds
# fresh keys - three thousand sentences of a five-satellite solution measured
# fifteen thousand entries, and that growth is linear in uptime, in a daemon
# meant to run for months.
GSA_EPOCH_ENTRY_LIMIT = 96

# NMEA 4.10 systemId, and the talker that names the same constellation
# everywhere else in the payload. An id outside this table still gets its own
# label rather than being folded in with its neighbours, because two systems
# this table does not know about are still two systems.
GSA_SYSTEM_TALKERS = {1: "GP", 2: "GL", 3: "GA", 4: "GB", 5: "GQ", 6: "GI"}

# Carrier-to-noise below ~10 dB-Hz is a satellite the receiver knows about but
# cannot demodulate. Counting those as "tracked" would make an antenna fault
# read as a healthy sky.
TRACKED_SNR_DB_HZ = 10

# The heartbeat is read by a phone over a tunnel. A multi-GNSS receiver with a
# clear sky can list well past 40 satellites, and nobody reads past the top of
# that list.
SKY_ENTRY_LIMIT = 40

# 99.99 is the receiver's saturation value for "no usable geometry", not a
# measurement. Rendered as a number it looks like catastrophically bad DOP and
# sends the reader hunting a geometry problem instead of reading "no fix".
DOP_SATURATION = 99.99

# GSA carries mode, fixType, twelve PRN slots and three DOPs. gps.py:239 guards
# GGA with `len(fields) >= 8` before indexing; the same floor here rejects a
# line truncated by a noisy cable before it can raise IndexError.
GSA_MINIMUM_FIELDS = 8

# GSV always carries total / index / in-view, even when in-view is zero and
# there are no satellite blocks at all.
GSV_MINIMUM_FIELDS = 3


def _optional_int(value: str) -> int | None:
    """
    Int, or None for a field the receiver left blank.

    Blank and zero are different answers here and the distinction is the whole
    point of the page: a blank SNR means "the almanac says it is up there and
    we hear nothing", while "00" means "we hear it at 0 dB-Hz". Collapsing
    blank to 0 throws away the strongest hardware evidence available.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_dop(value: str) -> float | None:
    """Dilution of precision, with the saturation value mapped to None."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if parsed >= DOP_SATURATION else parsed


def _snr_rank(snr: int | None) -> int:
    """Sort key that puts an unheard satellite below one heard at 0 dB-Hz."""
    return -1 if snr is None else snr


def _constellation_of(talker: str, system_id: int | None) -> str:
    """
    The talker that names the constellation a GSA is actually describing.

    Every $GNGSA in a 4.10 epoch carries the talker "GN" whatever constellation
    it lists, so tagging its PRNs with the sentence's talker turns GPS PRN 5 and
    Galileo PRN 5 into the same ("GN", 5) pair - collapsing precisely the
    cross-constellation collision the (talker, prn) identity exists to keep
    apart. Galileo PRNs run 1-36 in NMEA, so that collision is the normal case
    on a multi-GNSS receiver, and the payload then reports eleven satellites
    used on a twelve-satellite solution.

    Pre-4.10 sentences carry no systemId at all and separate by talker instead
    ($GPGSA, $GLGSA), so falling back to the sentence's own talker there is the
    receiver's own answer rather than a guess.
    """
    if system_id is None:
        return talker
    return GSA_SYSTEM_TALKERS.get(system_id) or f"{talker}{system_id}"


def parse_gsa(fields: list[str]) -> dict | None:
    """
    One GSA sentence: which satellites went into the solution, and how good it
    is. `fields` is already sliced (see the module docstring).

    Returns None for a sentence too short to trust. Note that systemId is not
    optional out of politeness - pre-NMEA-4.10 receivers emit several $GNGSA
    per epoch with no systemId field at all, so its absence is ordinary input
    and must not be read as a malformed line.
    """
    if len(fields) < GSA_MINIMUM_FIELDS:
        return None

    used_prns = []
    for prn_slot in fields[2:14]:
        prn = _optional_int(prn_slot)
        # Unused slots are blank, not zero, and a receiver holding a 5-satellite
        # solution leaves seven of the twelve empty.
        if prn is not None:
            used_prns.append(prn)

    return {
        "fix_type": _optional_int(fields[1]),
        "used_prns": used_prns,
        "pdop": _parse_dop(fields[14]) if len(fields) > 14 else None,
        "hdop": _parse_dop(fields[15]) if len(fields) > 15 else None,
        "vdop": _parse_dop(fields[16]) if len(fields) > 16 else None,
        "system_id": _optional_int(fields[17]) if len(fields) > 17 else None,
    }


def parse_gsv(fields: list[str]) -> dict | None:
    """
    One GSV sentence: up to four satellites the receiver believes are up there.
    `fields` is already sliced (see the module docstring).

    Returns None for a sentence whose header does not describe a coherent
    cycle. Refusing is the right answer: a GSV whose index exceeds its total
    cannot be placed in any cycle, and guessing where it belongs corrupts the
    in-view count rather than losing one sentence out of a stream that repeats
    every second.
    """
    if len(fields) < GSV_MINIMUM_FIELDS:
        return None

    total = _optional_int(fields[0])
    index = _optional_int(fields[1])
    in_view = _optional_int(fields[2])
    if total is None or index is None or in_view is None:
        return None
    if total < 1 or index < 1 or index > total:
        return None

    satellites = []
    block = 0
    # Consume only COMPLETE four-field blocks. This bound is the entire defence
    # against two separate wrong answers. NMEA 4.10 appends a lone signalId
    # after the last satellite, and a chunker that walks to len(fields) reads
    # that '1' as a fourth PRN and reports 12 in view on a sentence carrying
    # 11. In the other direction, the last sentence of a cycle carries 1-4
    # satellites rather than always 4, and assuming 4 fabricates a satellite
    # with every field blank - which then has no SNR and drags the tracked
    # count down, making a working antenna look partly dead.
    while 3 + 4 * block + 3 < len(fields):
        prn = _optional_int(fields[3 + 4 * block])
        elevation = _optional_int(fields[4 + 4 * block])
        azimuth = _optional_int(fields[5 + 4 * block])
        snr = _optional_int(fields[6 + 4 * block])
        # A satellite with no PRN is not a satellite. Elevation, azimuth and
        # SNR are all allowed to be blank on a satellite that is genuinely in
        # view, so none of them may gate the append.
        if prn is not None:
            satellites.append((prn, elevation, azimuth, snr))
        block += 1

    consumed = 3 + 4 * block
    # Exactly one leftover field is the NMEA 4.10 signalId. Anything else is a
    # ragged line, and inventing a signal id from it would split one
    # constellation's cycle across two accumulator keys - which shows up as an
    # in-view count that never completes.
    signal_id = fields[consumed] if len(fields) - consumed == 1 else ""

    return {
        "total": total,
        "index": index,
        "in_view": in_view,
        "signal_id": signal_id,
        "sats": satellites,
    }


@dataclass
class _GsaEpoch:
    """
    The reconciled GSA state for one epoch, where an epoch ends at the next RMC
    or, if that RMC is lost to a checksum failure, at the silence before the
    next burst.

    fix_type is the MAX across the epoch and never the last seen value. A
    multi-GNSS receiver emits one $GNGSA per constellation per epoch, and a
    constellation contributing nothing to the solution reports fixType '1'.
    Last-seen or min therefore reports "no fix" whenever any constellation is
    empty - which in a garage is most of them, permanently, and on the road is
    whichever constellation is currently behind a hill.
    """
    fix_type: int | None = None
    pdop: float | None = None
    hdop: float | None = None
    vdop: float | None = None
    # Keyed on (talker, tag, prn) so a PRN collision between two constellations
    # does not silently merge into one satellite. See _absorb_gsa for the tag.
    used: dict = field(default_factory=dict)
    sentences: int = 0
    # When the most recent GSA of this epoch was HEARD. It bounds the epoch on
    # its own, because the RMC that should close it is one checksum failure away
    # from never arriving.
    at: float = 0.0


class SkyView:
    """
    Accumulates GSA and GSV into the sky half of the heartbeat.

    Kept separate from NmeaAssembler because the two have opposite lifetimes:
    a Fix is a single instant, while the sky is a rolling window that has to
    survive across epochs and expire on its own when a constellation goes
    quiet.
    """

    def __init__(self) -> None:
        # GSV cycles in progress, and the last COMPLETE cycle, both keyed on
        # (talker, signal_id).
        #
        # Keying matters more than anything else in this class. GSV is the one
        # sentence family that does not collapse to the GN talker: a u-blox
        # emits $GPGSV, $GLGSV and $GAGSV as separate interleaved cycles and
        # never $GNGSV, the opposite of RMC/GGA/GSA. A single global partial
        # lets each constellation's index==1 wipe the previous constellation's
        # in-progress cycle, and the in-view count then silently reports only
        # whichever constellation was seen last.
        #
        # signal_id is in the key for a second, subtler reason: under NMEA
        # 4.10 one constellation runs two concurrent cycles, one per signal.
        # Keyed on talker alone the count oscillates - 5, 2, 5, 2 every second
        # - which reads as a flaky receiver and sends you hunting a hardware
        # fault that does not exist.
        self._partial: dict[tuple[str, str], dict] = {}
        self._published: dict[tuple[str, str], dict] = {}

        self._gsa_epoch: _GsaEpoch | None = None
        self._gsa_published: _GsaEpoch | None = None

        # HDOP is deliberately NOT part of snapshot(): the heartbeat's hdop
        # comes from GGA and belongs to Fix, and shipping two different HDOPs
        # under one name is how a page ends up disagreeing with itself. It is
        # exposed here so the epoch reconciliation is directly testable and so
        # a later reader is not tempted to re-derive it.
        self.gsa_hdop: float | None = None

    # -- ingest -----------------------------------------------------------
    def feed(self, kind: str, talker: str, fields: list[str], now: float) -> None:
        """Consume one already-sliced GSA or GSV sentence. Never raises."""
        talker = (talker or "").upper()
        if kind == "GSV":
            self._absorb_gsv(talker, fields, now)
        elif kind == "GSA":
            self._absorb_gsa(talker, fields, now)

    def _absorb_gsv(self, talker: str, fields: list[str], now: float) -> None:
        parsed = parse_gsv(fields)
        if parsed is None:
            return

        key = (talker, parsed["signal_id"])
        index, total = parsed["index"], parsed["total"]

        if index == 1:
            # A fresh index==1 abandons whatever was half-built. The previous
            # cycle either completed (and is already published) or was cut
            # short, and a cut-short cycle has no satellites worth keeping.
            partial = {
                "total": total,
                "next_index": 2,
                "in_view": parsed["in_view"],
                "sats": list(parsed["sats"]),
                # started_at is when the OLDEST satellite in this cycle was
                # heard and at is when the newest was. Both are needed: the
                # first decides when the published cycle goes stale, the second
                # decides whether a continuation still belongs here.
                "started_at": now,
                "at": now,
            }
            self._partial[key] = partial
        else:
            partial = self._partial.get(key)
            # Never guess at a gap. A sentence that does not continue exactly
            # the cycle we are holding could belong to a cycle whose earlier
            # sentences were lost to a checksum failure, and stitching it on
            # produces a satellite list that is a blend of two epochs.
            #
            # Age is part of "exactly". A partial with no age bound is worse
            # than a blend of two epochs: sentence 3 of a cycle an hour later
            # matches total and next_index perfectly, extends satellites nobody
            # has heard since, and republishes them with a fresh timestamp that
            # the staleness expiry then protects for another ten seconds.
            if (partial is None
                    or partial["total"] != total
                    or partial["next_index"] != index
                    or now - partial["at"] > PARTIAL_CYCLE_MAX_AGE_SECONDS):
                self._partial.pop(key, None)
                return
            partial["sats"].extend(parsed["sats"])
            partial["next_index"] = index + 1
            partial["at"] = now

        # A cycle is complete when and only when index == total. Publishing a
        # partial makes the in-view count sawtooth 4, 8, 11, 4, 8, 11 at 1Hz,
        # which on a live page is indistinguishable from satellites genuinely
        # appearing and vanishing.
        if index == total:
            self._published[key] = {
                "sats": list(partial["sats"]),
                "in_view": partial["in_view"],
                # Stamped with when the cycle STARTED, not with when it
                # completed. Expiry has to run from the moment the satellites
                # were heard, or a cycle assembled slowly out of stale
                # sentences buys itself a full fresh window on the page.
                "at": partial["started_at"],
            }
            self._partial.pop(key, None)

    def _absorb_gsa(self, talker: str, fields: list[str], now: float) -> None:
        parsed = parse_gsa(fields)
        if parsed is None:
            return

        epoch = self._gsa_epoch
        # An epoch whose closing RMC never arrived is abandoned rather than
        # extended. Merging the next burst into it reports the MAX fix_type and
        # the union of the PRNs across both, so a 3D fix from a second ago
        # survives beside a receiver that has since lost the sky entirely -
        # a held "3D fix" next to a void RMC, which is the one combination that
        # makes a reader distrust the whole page. The abandoned epoch is not
        # published: it was never closed, and a half-seen burst is exactly the
        # mid-burst reading on_rmc exists to prevent.
        if epoch is not None and now - epoch.at > GSA_EPOCH_MAX_AGE_SECONDS:
            epoch = None
        if epoch is None:
            epoch = _GsaEpoch(at=now)
            self._gsa_epoch = epoch

        # systemId separates the constellations under NMEA 4.10. Pre-4.10 there
        # is no such field, so we fall back to the sentence's arrival position
        # within the epoch - each constellation emits exactly one GSA per
        # epoch, which makes arrival order a sound stand-in for exactly as long
        # as the epoch lasts.
        tag = parsed["system_id"]
        if tag is None:
            tag = epoch.sentences
        epoch.sentences += 1
        epoch.at = now

        # The value carries the constellation the sentence DESCRIBES, not the
        # talker it was sent under; see _constellation_of for why "GN" on every
        # sentence is not an identity.
        constellation = _constellation_of(talker, parsed["system_id"])
        for prn in parsed["used_prns"]:
            identity = (talker, tag, prn)
            # A daemon whose receiver emits no RMC keeps one epoch open for as
            # long as it runs, and the arrival-order tag makes every pre-4.10
            # sentence contribute new keys. Refusing past the limit costs a
            # PRN or two on an epoch that is already nonsense and keeps a
            # month-long process from growing without bound.
            if identity not in epoch.used and len(epoch.used) >= GSA_EPOCH_ENTRY_LIMIT:
                continue
            epoch.used[identity] = (constellation, prn)

        fix_type = parsed["fix_type"]
        if fix_type is not None and (epoch.fix_type is None or fix_type > epoch.fix_type):
            epoch.fix_type = fix_type
            # DOPs describe the COMBINED solution, not the constellation, so
            # they are identical on every GSA that is actually contributing.
            # Taking them from the first sentence reaching the max avoids
            # inheriting the 99.99 saturation values that an empty
            # constellation's GSA carries.
            epoch.pdop = parsed["pdop"]
            epoch.hdop = parsed["hdop"]
            epoch.vdop = parsed["vdop"]

    def on_rmc(self) -> None:
        """
        Close the GSA epoch. The boundary is the next RMC rather than a
        wall-clock timer: a receiver emits its whole GSA burst and then the RMC,
        so RMC is the only marker that is guaranteed to fall between two bursts
        rather than through the middle of one. _absorb_gsa keeps a staleness
        backstop for the case where the RMC never arrives at all, but that
        backstop abandons its epoch rather than publishing it - only a real RMC
        says the burst was complete.
        """
        if self._gsa_epoch is not None:
            self._gsa_published = self._gsa_epoch
            self.gsa_hdop = self._gsa_epoch.hdop
        # An RMC with no GSA between it and the last one leaves the previous
        # epoch standing rather than blanking it. One dropped GSA would
        # otherwise flicker a held 3D fix to "unknown" on a page refreshing at
        # 2s; genuine loss of GSA is caught by the staleness expiry instead.
        self._gsa_epoch = None

    # -- publish ----------------------------------------------------------
    def snapshot(self, now: float) -> dict:
        """
        The sky half of the heartbeat payload: exactly these eight keys.

        satellites_used, hdop and fix_quality are absent on purpose - they come
        from GGA and stay on Fix.
        """
        self._expire(now)

        # Dedupe on (talker, prn) and never on bare prn. GPS PRN 5 and Galileo
        # PRN 5 are different satellites, and u-blox remaps PRNs into different
        # NMEA ranges depending on the configured NMEA version, so deduping on
        # the bare number silently undercounts on exactly the receivers that
        # can see the most.
        best_by_satellite: dict[tuple[str, int], dict] = {}
        for (talker, _signal_id), cycle in sorted(self._published.items()):
            for prn, elevation, azimuth, snr in cycle["sats"]:
                identity = (talker, prn)
                existing = best_by_satellite.get(identity)
                # The same satellite reported on two signals keeps the stronger
                # reading: tracked on L1 and silent on L5 is a tracked
                # satellite, and taking whichever arrived last would report it
                # as unheard half the time.
                if existing is None or _snr_rank(snr) > _snr_rank(existing["snr"]):
                    best_by_satellite[identity] = {
                        "talker": talker,
                        "prn": prn,
                        "elevation": elevation,
                        "azimuth": azimuth,
                        "snr": snr,
                    }

        constellations: dict[str, dict] = {}
        # Seed from the published keys rather than from the satellites, so a
        # constellation reporting a live and honest "00 in view" is still named
        # on the page. Silence and zero are different diagnoses.
        for talker, _signal_id in self._published:
            constellations.setdefault(
                talker, {"in_view": 0, "tracked": 0, "max_snr": None})

        for entry in best_by_satellite.values():
            bucket = constellations[entry["talker"]]
            bucket["in_view"] += 1
            snr = entry["snr"]
            if snr is not None and snr >= TRACKED_SNR_DB_HZ:
                bucket["tracked"] += 1
            if snr is not None and (bucket["max_snr"] is None or snr > bucket["max_snr"]):
                bucket["max_snr"] = snr

        sky = sorted(
            best_by_satellite.values(),
            key=lambda entry: (-_snr_rank(entry["snr"]), entry["prn"]),
        )[:SKY_ENTRY_LIMIT]

        tracked = sum(
            1 for entry in best_by_satellite.values()
            if entry["snr"] is not None and entry["snr"] >= TRACKED_SNR_DB_HZ
        )

        epoch = self._gsa_published
        return {
            "satellites_in_view": len(best_by_satellite),
            "satellites_tracked": tracked,
            "pdop": epoch.pdop if epoch else None,
            "vdop": epoch.vdop if epoch else None,
            "gsa_fix_type": epoch.fix_type if epoch else None,
            # Emitted as [talker, prn] pairs, deduped again at the boundary:
            # the (talker, tag, prn) union above exists to stop two
            # constellations merging inside one epoch, not to license the same
            # satellite appearing twice in the payload.
            "gsa_used_prns": [list(pair) for pair in sorted(set(epoch.used.values()))] if epoch else [],
            "constellations": constellations,
            "sky": sky,
        }

    def _expire(self, now: float) -> None:
        """
        Drop anything older than the expiry window.

        Without this a constellation that goes silent keeps its satellites on
        the page forever at their last-known SNR - the single most likely way
        this page tells a comfortable lie indoors, because the numbers it shows
        are the ones from the last time it worked.
        """
        for key in [key for key, cycle in self._published.items()
                    if now - cycle["at"] > PUBLISHED_CYCLE_EXPIRY_SECONDS]:
            del self._published[key]

        # Partials expire on the same bound that refuses to extend them. A
        # cycle that never completes is otherwise retained for the life of the
        # process, one entry per distinct (talker, signal_id) - small, but a
        # receiver that keeps changing signal ids grows the dict forever, and a
        # partial nothing may legally extend is dead weight regardless.
        for key in [key for key, partial in self._partial.items()
                    if now - partial["at"] > PARTIAL_CYCLE_MAX_AGE_SECONDS]:
            del self._partial[key]

        if (self._gsa_published is not None
                and now - self._gsa_published.at > PUBLISHED_CYCLE_EXPIRY_SECONDS):
            self._gsa_published = None
            self.gsa_hdop = None
