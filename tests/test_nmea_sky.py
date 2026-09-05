"""
test_nmea_sky.py

Behavioural tests for the GSA/GSV sky parser.

The dangerous failure here is not a crash. It is an inverted diagnosis: a
receiver whose antenna has fallen off reported as "healthy, just needs sky",
or a receiver holding a good 3D fix reported as "no fix" because one empty
constellation spoke last. Both are silent, both look entirely plausible on the
page, and both send someone to fix a thing that is not broken while the thing
that is broken stays broken. So these tests are mostly about counting: which
satellites, from which constellation, at which signal strength, and how long
after the receiver last mentioned them.

Every fixture below is a real NMEA sentence carrying its OWN published
checksum. None of them is built through a with_checksum() helper, for the
reason tests/test_gps.py:33-36 gives: a fixture that computes its own checksum
still passes when the XOR routine is wrong, so it proves nothing about the
bytes a receiver actually puts on the wire.

Run:  python3 tests/test_nmea_sky.py
"""
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import nmea_sky      # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures - real sentences, published checksums, never recomputed
# ---------------------------------------------------------------------------
# The off-by-one canary. Its four sub-fields are deliberately distinct in
# magnitude, so any one-field slip pushes 310 into elevation (which cannot
# exceed 90) or into SNR (which cannot exceed 99).
GSV_CANARY = "$GPGSV,1,1,01,07,72,310,41*4D"

# A three-sentence GPS cycle: 11 satellites in view at SNR 15-26. Paired with
# an indoor GSA below, this is the "antenna works, needs sky" case.
GSV_GPS_1_OF_3 = "$GPGSV,3,1,11,03,03,111,18,04,15,270,21,06,01,010,17,13,06,292,20*7A"
GSV_GPS_2_OF_3 = "$GPGSV,3,2,11,14,25,170,23,16,57,208,26,18,67,296,22,19,40,246,19*77"
# Sentence 3 of 3 carries three satellites, not four: 11 = 4 + 4 + 3.
GSV_GPS_3_OF_3 = "$GPGSV,3,3,11,22,42,067,24,24,14,311,16,27,05,244,15*49"
# The same last sentence with an NMEA 4.10 signalId glued on the end.
GSV_GPS_3_OF_3_WITH_SIGNAL_ID = "$GPGSV,3,3,11,22,42,067,24,24,14,311,16,27,05,244,15,1*54"

# A two-sentence GLONASS cycle, interleaved with the GPS one on the wire.
GSV_GLONASS_1_OF_2 = "$GLGSV,2,1,05,65,15,043,22,66,43,097,25,72,31,213,18,81,09,308,00*68"
GSV_GLONASS_2_OF_2 = "$GLGSV,2,2,05,82,27,145,21*5C"

# One Galileo constellation running two concurrent NMEA 4.10 signal cycles.
GSV_GALILEO_SIGNAL_7 = "$GAGSV,2,1,05,05,22,180,19,09,48,251,24,34,11,062,17,36,08,301,12,7*7D"
GSV_GALILEO_SIGNAL_2 = "$GAGSV,1,1,02,05,22,180,15,09,48,251,20,2*7D"

# The three shapes a receiver emits when it can see nothing at all.
GSV_ZERO_PRE_410 = "$GPGSV,1,1,00*79"
GSV_ZERO_SIGNAL_0 = "$GPGSV,1,1,00,0*65"
GSV_ZERO_SIGNAL_1 = "$GLGSV,1,1,00,1*78"

# Three satellites known from the stored almanac, every SNR field blank. This
# is what a disconnected antenna or a shielded car park looks like.
GSV_ANTENNA_DEAD = "$GPGSV,1,1,03,03,03,111,,04,15,270,,06,01,010,*48"

# GSA: a 3D fix from a pre-4.10 receiver, with no systemId field at all.
GSA_3D_PRE_410 = "$GPGSA,A,3,04,05,,09,12,,,24,,,,,2.5,1.3,2.1*39"
# Indoors: no fix, and every DOP saturated at 99.99.
GSA_NO_FIX_PRE_410 = "$GNGSA,A,1,,,,,,,,,,,,,99.99,99.99,99.99*2E"
# The max-not-last-seen pair: GPS holds a 3D fix, GLONASS contributes nothing
# and honestly reports fixType 1.
GSA_EPOCH_GPS_3D = "$GNGSA,A,3,03,04,06,09,,,,,,,,,2.10,1.10,1.80,1*03"
GSA_EPOCH_GLONASS_EMPTY = "$GNGSA,A,1,,,,,,,,,,,,,99.99,99.99,99.99,2*30"
# An NMEA 4.10 epoch where both constellations contribute.
GSA_410_GPS = "$GNGSA,A,3,03,04,06,09,17,19,22,28,,,,,1.72,0.95,1.43,1*03"
GSA_410_GLONASS = "$GNGSA,A,3,65,66,72,81,,,,,,,,,1.72,0.95,1.43,2*03"
# The same epoch indoors.
GSA_410_INDOOR_GPS = "$GNGSA,A,1,,,,,,,,,,,,,99.99,99.99,99.99,1*33"
GSA_410_INDOOR_GLONASS = "$GNGSA,A,1,,,,,,,,,,,,,99.99,99.99,99.99,2*30"

# Deduplicated: the indoor 4.10 pair's GLONASS half is byte-identical to the
# empty-constellation half of the max-not-last-seen pair, because it is the
# same sentence a receiver emits in both situations.
ALL_FIXTURES = sorted({
    value for name, value in globals().items()
    if name.startswith(("GSV_", "GSA_")) and isinstance(value, str)
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# The XOR is reimplemented here rather than imported from src.gps on purpose.
# nmea_sky imports nothing from gps so that the two halves can fail
# independently, and a test that reaches into gps to check its own fixtures
# gives that up - a broken gps.py would then make the sky tests unrunnable
# rather than merely making the gps tests fail.
def checksum_matches_published(sentence: str) -> bool:
    """XOR of every byte between '$' and '*', against the sentence's own tail."""
    body, _, given = sentence.strip()[1:].partition("*")
    computed = 0
    for character in body:
        computed ^= ord(character)
    return f"{computed:02X}" == given[:2].upper()


def sliced(sentence: str) -> list[str]:
    """
    Slice exactly the way src/gps.py does before calling us. The trailing [1:]
    drops the "$GPGSV" token, which is the whole reason the field indices in
    nmea_sky are one lower than every NMEA reference on the internet.
    """
    return sentence.strip().split("*")[0].split(",")[1:]


def talker_of(sentence: str) -> str:
    return sentence.strip().lstrip("$")[0:2].upper()


def kind_of(sentence: str) -> str:
    return sentence.strip().lstrip("$")[2:5].upper()


def feed(sky: "nmea_sky.SkyView", sentence: str, now: float = 100.0) -> None:
    sky.feed(kind_of(sentence), talker_of(sentence), sliced(sentence), now)


def prns_for(snapshot: dict, talker: str) -> list[int]:
    return sorted(entry["prn"] for entry in snapshot["sky"] if entry["talker"] == talker)


def satellite_in(snapshot: dict, talker: str, prn: int) -> dict | None:
    for entry in snapshot["sky"]:
        if entry["talker"] == talker and entry["prn"] == prn:
            return entry
    return None


# ---------------------------------------------------------------------------
# The fixtures themselves
# ---------------------------------------------------------------------------
def test_every_fixture_carries_a_valid_published_checksum():
    # If this fails, every assertion below is being made against a sentence no
    # receiver would ever emit, and the suite is testing nothing.
    assert len(ALL_FIXTURES) == 20, f"expected 20 fixtures, found {len(ALL_FIXTURES)}"
    for sentence in ALL_FIXTURES:
        assert checksum_matches_published(sentence), sentence


# ---------------------------------------------------------------------------
# The field-index off-by-one
# ---------------------------------------------------------------------------
def test_snr_is_read_from_the_snr_field_and_not_from_the_azimuth():
    # Reading SNR one field early gives it the azimuth, which is 000-359 and
    # almost never zero. Every satellite then looks strongly received, so a
    # receiver with no antenna attached is reported as healthy - the exact
    # inversion of the answer this page exists to give.
    parsed = nmea_sky.parse_gsv(sliced(GSV_CANARY))
    assert parsed is not None
    prn, elevation, azimuth, snr = parsed["sats"][0]
    assert prn == 7, f"prn read as {prn}"
    assert elevation == 72, f"elevation read as {elevation}"
    assert azimuth == 310, f"azimuth read as {azimuth}"
    assert snr == 41, f"snr read as {snr}"


def test_a_gsa_reads_its_dops_from_the_dop_fields():
    # The same slip on GSA silently swaps PDOP for the twelfth PRN slot, which
    # is blank on most fixes - so the page would report "no geometry data" on
    # a receiver reporting perfect geometry.
    parsed = nmea_sky.parse_gsa(sliced(GSA_3D_PRE_410))
    assert parsed is not None
    assert parsed["fix_type"] == 3
    assert parsed["used_prns"] == [4, 5, 9, 12, 24], parsed["used_prns"]
    assert (parsed["pdop"], parsed["hdop"], parsed["vdop"]) == (2.5, 1.3, 2.1)
    assert parsed["system_id"] is None, "invented a systemId a pre-4.10 receiver never sent"


# ---------------------------------------------------------------------------
# Block iteration
# ---------------------------------------------------------------------------
def test_a_trailing_signal_id_is_not_counted_as_a_fourth_satellite():
    # Sliced length is 16, the last complete block ends at index 14, and
    # sliced[15] is the signalId '1'. A chunker that walks to len(fields) reads
    # it as a fourth PRN and reports 12 satellites in view on a cycle carrying
    # 11 - a permanent, invisible over-count.
    parsed = nmea_sky.parse_gsv(sliced(GSV_GPS_3_OF_3_WITH_SIGNAL_ID))
    assert parsed is not None
    assert len(parsed["sats"]) == 3, parsed["sats"]
    assert [satellite[0] for satellite in parsed["sats"]] == [22, 24, 27]
    assert parsed["signal_id"] == "1"


def test_a_short_final_sentence_is_not_padded_to_four_satellites():
    # 11 in view over three sentences is 4 + 4 + 3. Assuming four fabricates a
    # satellite with every field blank, which then counts as in-view-but-not-
    # tracked and makes a working antenna look partly dead.
    parsed = nmea_sky.parse_gsv(sliced(GSV_GPS_3_OF_3))
    assert parsed is not None
    assert len(parsed["sats"]) == 3, parsed["sats"]
    assert parsed["signal_id"] == "", "read a satellite field as a signalId"


def test_all_three_zero_in_view_shapes_parse_without_raising():
    # A receiver 90 seconds into a cold start emits these, and an IndexError
    # here kills the reader thread that feeds the whole heartbeat - so the page
    # would report "service dead" for what is a completely normal receiver.
    for sentence in (GSV_ZERO_PRE_410, GSV_ZERO_SIGNAL_0, GSV_ZERO_SIGNAL_1):
        parsed = nmea_sky.parse_gsv(sliced(sentence))
        assert parsed is not None, sentence
        assert parsed["in_view"] == 0, sentence
        assert parsed["sats"] == [], sentence

        sky = nmea_sky.SkyView()
        feed(sky, sentence)
        snapshot = sky.snapshot(100.0)
        assert snapshot["satellites_in_view"] == 0, sentence
        assert snapshot["sky"] == [], sentence
        # The constellation is still named, because "GPS reports zero in view"
        # and "GPS has said nothing at all" are different diagnoses.
        assert talker_of(sentence) in snapshot["constellations"], sentence


def test_malformed_sentences_are_refused_rather_than_guessed_at():
    # Rubbish arrives on a noisy 5m USB extension. Returning None loses one
    # sentence out of a stream that repeats every second; guessing corrupts the
    # count until the next power cycle.
    for fields in ([], [""], ["1", "1"], ["x", "y", "z"], ["1", "5", "04"]):
        assert nmea_sky.parse_gsv(fields) is None, fields
    for fields in ([], ["A"], ["A", "3", "04"]):
        assert nmea_sky.parse_gsa(fields) is None, fields


# ---------------------------------------------------------------------------
# Multi-sentence accumulation
# ---------------------------------------------------------------------------
def test_interleaved_gps_and_glonass_cycles_do_not_overwrite_each_other():
    # GSV is the one family that never collapses to the GN talker: a u-blox
    # interleaves $GPGSV and $GLGSV cycles. An accumulator keyed globally lets
    # each constellation's index==1 wipe the other's in-progress cycle, and the
    # page then reports 5 satellites in view when 16 are up there.
    sky = nmea_sky.SkyView()
    for sentence in (GSV_GPS_1_OF_3, GSV_GLONASS_1_OF_2, GSV_GPS_2_OF_3,
                     GSV_GLONASS_2_OF_2, GSV_GPS_3_OF_3):
        feed(sky, sentence)

    snapshot = sky.snapshot(100.0)
    assert snapshot["satellites_in_view"] == 16, snapshot["satellites_in_view"]
    assert prns_for(snapshot, "GP") == [3, 4, 6, 13, 14, 16, 18, 19, 22, 24, 27]
    assert len(prns_for(snapshot, "GL")) == 5
    # Per-satellite SNR must survive the interleave intact, or the tracked
    # count is right for the wrong reasons.
    assert satellite_in(snapshot, "GL", 65)["snr"] == 22
    assert satellite_in(snapshot, "GL", 81)["snr"] == 0
    assert snapshot["constellations"]["GP"]["in_view"] == 11
    assert snapshot["constellations"]["GL"]["max_snr"] == 25


def test_a_partial_cycle_is_never_published():
    # Publishing on every sentence makes the in-view count sawtooth 4, 8, 11 at
    # 1Hz, which on a live page is indistinguishable from satellites genuinely
    # appearing and vanishing - and sends the reader looking for interference.
    sky = nmea_sky.SkyView()
    feed(sky, GSV_GPS_1_OF_3)
    assert sky.snapshot(100.0)["satellites_in_view"] == 0
    feed(sky, GSV_GPS_2_OF_3)
    assert sky.snapshot(100.0)["satellites_in_view"] == 0
    feed(sky, GSV_GPS_3_OF_3)
    assert sky.snapshot(100.0)["satellites_in_view"] == 11


def test_a_cycle_with_a_missing_middle_sentence_is_discarded_whole():
    # Sentence 2 was lost to a checksum failure. Stitching sentence 3 onto
    # sentence 1 publishes a satellite list blended from two different epochs,
    # which is worse than publishing nothing for one second.
    sky = nmea_sky.SkyView()
    feed(sky, GSV_GPS_1_OF_3)
    feed(sky, GSV_GPS_3_OF_3)
    assert sky.snapshot(100.0)["satellites_in_view"] == 0

    # And the receiver recovers on the next complete cycle rather than staying
    # wedged.
    for sentence in (GSV_GPS_1_OF_3, GSV_GPS_2_OF_3, GSV_GPS_3_OF_3):
        feed(sky, sentence)
    assert sky.snapshot(100.0)["satellites_in_view"] == 11


def test_two_signal_cycles_from_one_constellation_are_kept_apart():
    # Both of these are $GAGSV from the same receiver in the same second: one
    # signalId 7 reporting 5 in view, one signalId 2 reporting 2. Keyed on the
    # talker alone the count oscillates 5, 2, 5, 2 every second, which reads as
    # a flaky receiver and sends you hunting a hardware fault that is not there.
    signal_7 = nmea_sky.parse_gsv(sliced(GSV_GALILEO_SIGNAL_7))
    signal_2 = nmea_sky.parse_gsv(sliced(GSV_GALILEO_SIGNAL_2))
    assert signal_7["signal_id"] == "7" and signal_7["in_view"] == 5
    assert signal_2["signal_id"] == "2" and signal_2["in_view"] == 2
    assert len(signal_7["sats"]) == 4 and len(signal_2["sats"]) == 2

    sky = nmea_sky.SkyView()
    feed(sky, GSV_GALILEO_SIGNAL_7)     # 1 of 2, stays in progress
    feed(sky, GSV_GALILEO_SIGNAL_2)     # 1 of 1, completes immediately
    assert sky.snapshot(100.0)["satellites_in_view"] == 2

    # The accumulator key is inspected directly because the contract supplies
    # no second $GAGSV for signal 7, so there is no way to watch that cycle
    # complete through snapshot() alone - and "did the signal-2 cycle destroy
    # the signal-7 one" is exactly the fact under test.
    assert ("GA", "7") in sky._partial, "the signal 7 cycle was wiped by signal 2"
    assert ("GA", "2") in sky._published


def test_a_fresh_cycle_replaces_the_previous_one_rather_than_accumulating():
    # Satellites set. A cycle that merged into its predecessor would grow the
    # in-view count monotonically all day, so a car parked overnight would wake
    # up claiming to see thirty GPS satellites - there are thirty-one in orbit
    # and roughly a third are ever above the horizon.
    sky = nmea_sky.SkyView()
    for sentence in (GSV_GPS_1_OF_3, GSV_GPS_2_OF_3, GSV_GPS_3_OF_3):
        feed(sky, sentence)
    assert sky.snapshot(100.0)["satellites_in_view"] == 11

    feed(sky, GSV_ANTENNA_DEAD)
    snapshot = sky.snapshot(100.0)
    assert snapshot["satellites_in_view"] == 3, snapshot["satellites_in_view"]
    assert prns_for(snapshot, "GP") == [3, 4, 6]


def test_the_same_prn_from_two_constellations_counts_as_two_satellites():
    # GPS PRN 5 and Galileo PRN 5 are different satellites, and u-blox remaps
    # PRNs into different NMEA ranges depending on the configured NMEA version.
    # Deduping on the bare number silently undercounts on exactly the receivers
    # that can see the most. The same fields are fed under two talkers because
    # that is precisely the variable under test.
    sky = nmea_sky.SkyView()
    sky.feed("GSV", "GP", sliced(GSV_CANARY), 100.0)
    sky.feed("GSV", "GL", sliced(GSV_CANARY), 100.0)

    snapshot = sky.snapshot(100.0)
    assert snapshot["satellites_in_view"] == 2, snapshot["satellites_in_view"]
    assert satellite_in(snapshot, "GP", 7) is not None
    assert satellite_in(snapshot, "GL", 7) is not None


def test_a_stale_partial_is_not_stitched_onto_a_sentence_from_a_later_cycle():
    # Sentences 1 and 2 land, sentence 3 is lost to a checksum failure - which
    # gps.py drops silently, so it is a routine event. Eighty-two minutes later
    # sentence 3 of a NEW cycle arrives, and it matches the held partial's total
    # and next index exactly. Extending it republishes eleven satellites at SNR
    # 15-26 with a fresh timestamp, which the ten-second expiry then guards: the
    # page shows a healthy sky assembled from satellites nobody has heard since
    # the car was parked, which is the precise lie this module exists to prevent.
    sky = nmea_sky.SkyView()
    feed(sky, GSV_GPS_1_OF_3, now=100.0)
    feed(sky, GSV_GPS_2_OF_3, now=100.0)

    feed(sky, GSV_GPS_3_OF_3, now=5000.0)
    snapshot = sky.snapshot(5000.0)
    assert snapshot["satellites_in_view"] == 0, snapshot["satellites_in_view"]
    assert snapshot["sky"] == [], snapshot["sky"]
    assert sky._partial == {}, sky._partial

    # And a cycle that completes at the normal cadence is untouched by the bound.
    feed(sky, GSV_GPS_1_OF_3, now=5001.0)
    feed(sky, GSV_GPS_2_OF_3, now=5001.0)
    feed(sky, GSV_GPS_3_OF_3, now=5001.0)
    assert sky.snapshot(5001.0)["satellites_in_view"] == 11


def test_a_published_cycle_expires_from_when_its_satellites_were_heard():
    # Expiry measured from publication rather than from reception lets a cycle
    # assembled out of old sentences buy itself a full fresh ten-second window,
    # so the oldest satellite on the page can be arbitrarily older than the page
    # admits. This cycle's first satellites were heard at t=1000, so all eleven
    # are stale at t=1011 even though the cycle only completed at t=1002.
    sky = nmea_sky.SkyView()
    feed(sky, GSV_GPS_1_OF_3, now=1000.0)
    feed(sky, GSV_GPS_2_OF_3, now=1001.0)
    feed(sky, GSV_GPS_3_OF_3, now=1002.0)
    assert sky.snapshot(1002.0)["satellites_in_view"] == 11

    assert sky.snapshot(1009.0)["satellites_in_view"] == 11
    assert sky.snapshot(1011.0)["satellites_in_view"] == 0, "a stale cycle lingered"


def test_a_cycle_that_never_completes_is_not_retained_forever():
    # One entry per distinct (talker, signal_id) is small, but a receiver whose
    # signal ids change with configuration adds keys that nothing will ever
    # remove, and a partial too old for any sentence to legally extend is dead
    # weight in a process that runs for months.
    sky = nmea_sky.SkyView()
    feed(sky, GSV_GPS_1_OF_3, now=100.0)
    assert ("GP", "") in sky._partial

    sky.snapshot(200.0)
    assert sky._partial == {}, sky._partial


def test_a_silent_constellation_drops_off_the_page_after_ten_seconds():
    # Without expiry, a constellation that goes quiet keeps its satellites on
    # the page forever at their last-known SNR. That is the single most likely
    # way this page tells a comfortable lie indoors: the numbers it shows are
    # the ones from the last time it worked.
    sky = nmea_sky.SkyView()
    for sentence in (GSV_GPS_1_OF_3, GSV_GLONASS_1_OF_2, GSV_GPS_2_OF_3,
                     GSV_GLONASS_2_OF_2, GSV_GPS_3_OF_3):
        feed(sky, sentence, now=1000.0)
    assert sky.snapshot(1000.0)["satellites_in_view"] == 16

    # Eleven seconds later GPS is still talking and GLONASS has gone silent.
    for sentence in (GSV_GPS_1_OF_3, GSV_GPS_2_OF_3, GSV_GPS_3_OF_3):
        feed(sky, sentence, now=1011.0)

    snapshot = sky.snapshot(1011.0)
    assert snapshot["satellites_in_view"] == 11, snapshot["satellites_in_view"]
    assert "GL" not in snapshot["constellations"], "a silent constellation lingered"
    assert prns_for(snapshot, "GL") == []


# ---------------------------------------------------------------------------
# GSA reconciliation
# ---------------------------------------------------------------------------
def test_gsa_fix_type_is_the_max_across_the_epoch_not_the_last_one_seen():
    # A multi-GNSS receiver emits one $GNGSA per constellation, and a
    # constellation contributing nothing reports fixType 1 with saturated DOPs.
    # Last-seen or min therefore reports "no fix" whenever any constellation is
    # empty - which in a garage is most of them, permanently, and while driving
    # is whichever one is behind a hill.
    sky = nmea_sky.SkyView()
    feed(sky, GSA_EPOCH_GPS_3D)
    feed(sky, GSA_EPOCH_GLONASS_EMPTY)
    sky.on_rmc()

    snapshot = sky.snapshot(100.0)
    assert snapshot["gsa_fix_type"] == 3, snapshot["gsa_fix_type"]
    assert sky.gsa_hdop == 1.10, sky.gsa_hdop
    assert snapshot["pdop"] == 2.10, snapshot["pdop"]
    assert snapshot["vdop"] == 1.80, snapshot["vdop"]


def test_a_saturated_dop_is_reported_as_no_value_rather_than_99_99():
    # Rendered as a number, 99.99 looks like catastrophically bad geometry and
    # sends the reader hunting a sky-view problem on a receiver that simply has
    # no solution at all.
    parsed = nmea_sky.parse_gsa(sliced(GSA_NO_FIX_PRE_410))
    assert parsed is not None
    assert parsed["fix_type"] == 1
    assert parsed["used_prns"] == []
    assert (parsed["pdop"], parsed["hdop"], parsed["vdop"]) == (None, None, None)


def test_a_410_epoch_unions_the_used_prns_from_every_constellation():
    # Each $GNGSA lists only its own constellation's contribution. Taking the
    # last one reports 4 satellites used on a 12-satellite solution, which
    # reads as marginal geometry on a fix that is actually excellent.
    sky = nmea_sky.SkyView()
    feed(sky, GSA_410_GPS)
    feed(sky, GSA_410_GLONASS)
    sky.on_rmc()

    snapshot = sky.snapshot(100.0)
    assert snapshot["gsa_fix_type"] == 3
    assert len(snapshot["gsa_used_prns"]) == 12, snapshot["gsa_used_prns"]
    # Tagged by the systemId the sentence carries, not by the "GN" talker both
    # sentences were sent under.
    assert ["GP", 3] in snapshot["gsa_used_prns"], snapshot["gsa_used_prns"]
    assert ["GL", 81] in snapshot["gsa_used_prns"], snapshot["gsa_used_prns"]
    assert sky.gsa_hdop == 0.95


def test_an_indoor_410_epoch_reports_no_fix_and_no_geometry():
    sky = nmea_sky.SkyView()
    feed(sky, GSA_410_INDOOR_GPS)
    feed(sky, GSA_410_INDOOR_GLONASS)
    sky.on_rmc()

    snapshot = sky.snapshot(100.0)
    assert snapshot["gsa_fix_type"] == 1
    assert snapshot["gsa_used_prns"] == []
    assert snapshot["pdop"] is None and snapshot["vdop"] is None


def test_gsa_state_is_published_only_at_the_epoch_boundary():
    # The boundary is the next RMC, not a wall-clock timer: a receiver emits
    # its whole GSA burst and then the RMC, so RMC is the only marker
    # guaranteed to fall between two bursts rather than through the middle of
    # one. Reporting mid-burst shows the GPS-only fix type for part of a second.
    sky = nmea_sky.SkyView()
    feed(sky, GSA_EPOCH_GPS_3D)
    assert sky.snapshot(100.0)["gsa_fix_type"] is None, "reported a half-built epoch"
    sky.on_rmc()
    assert sky.snapshot(100.0)["gsa_fix_type"] == 3


def test_a_receiver_that_stops_sending_gsa_stops_claiming_a_fix():
    # Same lie as a lingering constellation, but worse: a frozen "3D fix" beside
    # a void RMC is the one combination guaranteed to make a reader distrust the
    # whole page.
    sky = nmea_sky.SkyView()
    feed(sky, GSA_EPOCH_GPS_3D, now=1000.0)
    sky.on_rmc()
    assert sky.snapshot(1000.0)["gsa_fix_type"] == 3
    assert sky.snapshot(1011.0)["gsa_fix_type"] is None


def test_gsa_prns_are_tagged_by_the_system_they_describe_not_by_the_gn_talker():
    # Every $GNGSA of a 4.10 epoch carries the talker "GN" whichever
    # constellation it lists, so tagging its PRNs with the sentence talker makes
    # GPS PRN 3 and Galileo PRN 3 the same ("GN", 3) pair - and the union at the
    # payload boundary then reports eight satellites used on a sixteen-satellite
    # solution, which reads as marginal geometry on an excellent fix. Galileo
    # PRNs run 1-36 in NMEA, so this collision is the ordinary case.
    #
    # The systemId of a real fixture is edited rather than a second sentence
    # invented, for the same reason GSV_CANARY is fed under two talkers below:
    # systemId is precisely the variable under test, and everything else about
    # the two sentences must stay identical for the test to mean anything.
    galileo_fields = sliced(GSA_410_GPS)
    assert galileo_fields[-1] == "1", galileo_fields
    galileo_fields[-1] = "3"

    sky = nmea_sky.SkyView()
    feed(sky, GSA_410_GPS)
    sky.feed("GSA", "GN", galileo_fields, 100.0)
    sky.on_rmc()

    snapshot = sky.snapshot(100.0)
    assert len(snapshot["gsa_used_prns"]) == 16, snapshot["gsa_used_prns"]
    assert ["GP", 3] in snapshot["gsa_used_prns"], snapshot["gsa_used_prns"]
    assert ["GA", 3] in snapshot["gsa_used_prns"], snapshot["gsa_used_prns"]
    assert not any(pair[0] == "GN" for pair in snapshot["gsa_used_prns"]), \
        snapshot["gsa_used_prns"]


def test_a_pre_410_gsa_is_tagged_by_its_own_talker():
    # Pre-4.10 receivers send no systemId at all and separate their
    # constellations by talker instead. Treating that absence as a malformed
    # line, or tagging every such sentence alike, would lose the only
    # constellation identity those receivers ever give.
    sky = nmea_sky.SkyView()
    feed(sky, GSA_3D_PRE_410)
    sky.on_rmc()

    snapshot = sky.snapshot(100.0)
    assert snapshot["gsa_used_prns"] == [
        ["GP", 4], ["GP", 5], ["GP", 9], ["GP", 12], ["GP", 24],
    ], snapshot["gsa_used_prns"]


def test_an_unclosed_gsa_epoch_does_not_merge_into_the_next_burst():
    # The RMC that should have closed the first epoch was lost to a checksum
    # failure, which gps.py drops silently and is therefore a normal event. With
    # nothing but RMC to end an epoch, the next burst joins it: fix_type is a MAX
    # across the epoch, so the 3D fix from a second ago outranks the "no fix" the
    # receiver is reporting now and stays on the page beside a void RMC.
    sky = nmea_sky.SkyView()
    feed(sky, GSA_EPOCH_GPS_3D, now=100.0)
    # No on_rmc here - that is the whole scenario.
    feed(sky, GSA_410_INDOOR_GPS, now=101.0)
    sky.on_rmc()

    snapshot = sky.snapshot(101.0)
    assert snapshot["gsa_fix_type"] == 1, snapshot["gsa_fix_type"]
    assert snapshot["gsa_used_prns"] == [], snapshot["gsa_used_prns"]
    assert snapshot["pdop"] is None and snapshot["vdop"] is None


def test_one_burst_is_never_split_across_two_epochs():
    # The other side of the bound above. A burst arrives in a single read off
    # CDC-ACM, so its sentences are parsed milliseconds apart; splitting one
    # would report the GPS-only half of a solution and undo the max-not-last-seen
    # reconciliation this class exists for.
    sky = nmea_sky.SkyView()
    feed(sky, GSA_410_GPS, now=100.0)
    feed(sky, GSA_410_GLONASS, now=100.02)
    sky.on_rmc()

    snapshot = sky.snapshot(100.02)
    assert snapshot["gsa_fix_type"] == 3
    assert len(snapshot["gsa_used_prns"]) == 12, snapshot["gsa_used_prns"]


def test_a_receiver_that_never_sends_rmc_does_not_grow_the_epoch_without_bound():
    # A receiver configured with RMC disabled holds one epoch open for the life
    # of the daemon, and a pre-4.10 sentence is tagged by arrival order, so every
    # sentence contributes brand new keys: 3000 sentences measured 9000 entries,
    # roughly a quarter of a million a day at 1Hz in a process meant to run for
    # months.
    sky = nmea_sky.SkyView()
    for sentence_number in range(3000):
        # A tenth of a second apart, so the epoch's own staleness bound never
        # fires and the accumulation cap is the only thing under test.
        feed(sky, GSA_3D_PRE_410, now=100.0 + sentence_number * 0.1)

    assert sky._gsa_epoch is not None
    assert len(sky._gsa_epoch.used) <= nmea_sky.GSA_EPOCH_ENTRY_LIMIT, \
        len(sky._gsa_epoch.used)


# ---------------------------------------------------------------------------
# The two states this whole module exists to separate
# ---------------------------------------------------------------------------
def test_an_antenna_that_works_but_has_no_sky_reports_satellites_it_can_hear():
    # This is the parking-garage answer, and it must not read as an error. The
    # receiver hears 11 satellites perfectly well and simply has no solution.
    sky = nmea_sky.SkyView()
    for sentence in (GSV_GPS_1_OF_3, GSV_GPS_2_OF_3, GSV_GPS_3_OF_3):
        feed(sky, sentence)
    feed(sky, GSA_NO_FIX_PRE_410)
    sky.on_rmc()

    snapshot = sky.snapshot(100.0)
    assert snapshot["satellites_in_view"] == 11
    assert snapshot["satellites_tracked"] == 11, snapshot["satellites_tracked"]
    assert snapshot["gsa_used_prns"] == []
    assert snapshot["gsa_fix_type"] == 1


def test_a_dead_antenna_reports_satellites_in_view_that_it_cannot_hear():
    # "In view" comes from the stored almanac and needs no reception at all, so
    # in-view alone is never evidence of a working antenna. Three satellites
    # with every SNR field blank is the strongest hardware signal this page has,
    # and a parser that skips blank-SNR satellites destroys exactly that
    # evidence - the page would then show "0 in view", which is the cold-start
    # answer, not the antenna-fault answer.
    sky = nmea_sky.SkyView()
    feed(sky, GSV_ANTENNA_DEAD)
    feed(sky, GSA_NO_FIX_PRE_410)
    sky.on_rmc()

    snapshot = sky.snapshot(100.0)
    assert snapshot["satellites_in_view"] == 3, snapshot["satellites_in_view"]
    assert snapshot["satellites_tracked"] == 0, snapshot["satellites_tracked"]
    assert snapshot["gsa_used_prns"] == []
    assert all(entry["snr"] is None for entry in snapshot["sky"]), snapshot["sky"]
    assert snapshot["constellations"]["GP"]["max_snr"] is None


def test_a_satellite_heard_at_zero_is_not_the_same_as_one_not_heard_at_all():
    # Blank means "the almanac says it is up there and we hear nothing"; "00"
    # means "we hear it at 0 dB-Hz". Collapsing blank to 0 makes an antenna
    # fault and a satellite low on the horizon look identical.
    sky = nmea_sky.SkyView()
    feed(sky, GSV_GLONASS_1_OF_2)
    feed(sky, GSV_GLONASS_2_OF_2)
    feed(sky, GSV_ANTENNA_DEAD)

    snapshot = sky.snapshot(100.0)
    assert satellite_in(snapshot, "GL", 81)["snr"] == 0
    assert satellite_in(snapshot, "GP", 3)["snr"] is None
    # Neither counts as tracked, but both count as in view.
    assert snapshot["satellites_in_view"] == 8
    assert snapshot["satellites_tracked"] == 4, snapshot["satellites_tracked"]


# ---------------------------------------------------------------------------
# The payload boundary
# ---------------------------------------------------------------------------
def test_snapshot_ships_exactly_the_agreed_keys():
    # ui_app splices this straight into the heartbeat. An extra key here would
    # silently shadow one of GGA's, and a missing key is a KeyError on the page
    # rather than a null.
    sky = nmea_sky.SkyView()
    feed(sky, GSV_GPS_1_OF_3)
    snapshot = sky.snapshot(100.0)
    assert set(snapshot) == {
        "satellites_in_view", "satellites_tracked", "pdop", "vdop",
        "gsa_fix_type", "gsa_used_prns", "constellations", "sky",
    }, sorted(snapshot)
    # hdop and fix_quality come from GGA and belong to Fix. Two different HDOPs
    # under one name is how a page ends up disagreeing with itself.
    for forbidden in ("satellites_used", "hdop", "fix_quality"):
        assert forbidden not in snapshot, forbidden


def test_the_sky_list_is_ordered_by_signal_strength_with_the_unheard_last():
    # The page draws C/N0 bars from this list and truncates it. Unsorted, the
    # top of the list is whichever constellation happened to speak first, so
    # the strongest satellites can be the ones cut off.
    sky = nmea_sky.SkyView()
    for sentence in (GSV_GLONASS_1_OF_2, GSV_GLONASS_2_OF_2, GSV_ANTENNA_DEAD):
        feed(sky, sentence)

    snapshot = sky.snapshot(100.0)
    ranks = [-1 if entry["snr"] is None else entry["snr"] for entry in snapshot["sky"]]
    assert ranks == sorted(ranks, reverse=True), ranks
    assert snapshot["sky"][0]["snr"] == 25, snapshot["sky"][0]
    # An unheard satellite sorts below one heard at 0 dB-Hz, so the bars the
    # page can actually draw are never the ones that get truncated away.
    assert snapshot["sky"][-1]["snr"] is None, snapshot["sky"][-1]
    assert len(snapshot["sky"]) <= nmea_sky.SKY_ENTRY_LIMIT


def test_an_empty_sky_view_reports_nulls_rather_than_zeros_it_cannot_justify():
    # Before the first sentence there is no evidence of anything. Reporting a
    # fix type of 1 here would claim the receiver has said "no fix" when it has
    # said nothing at all - the difference between states 4 and 5 on the page.
    snapshot = nmea_sky.SkyView().snapshot(100.0)
    assert snapshot["satellites_in_view"] == 0
    assert snapshot["satellites_tracked"] == 0
    assert snapshot["gsa_fix_type"] is None
    assert snapshot["pdop"] is None and snapshot["vdop"] is None
    assert snapshot["constellations"] == {}
    assert snapshot["sky"] == []
    assert snapshot["gsa_used_prns"] == []


# ---------------------------------------------------------------------------
def main() -> int:
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as error:
            failures += 1
            print(f"  FAIL  {test.__name__}: {error}")
        except Exception as error:                        # noqa: BLE001
            failures += 1
            print(f"  ERROR {test.__name__}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
