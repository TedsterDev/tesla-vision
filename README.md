# tesla-vision

Counter-surveillance for a Tesla, running on a Jetson Orin Nano wired into the
car's USB port. The Jetson presents itself to the vehicle as the TeslaCam flash
drive, so it receives every Dashcam and Sentry clip the car records, then
analyses that footage for signs that a particular vehicle or person keeps
turning up wherever you go.

This is a port of [tevora-threat/Scout](https://github.com/tevora-threat/Scout)
(DEF CON 27, "Surveillance Detection Scout - Your Lookout on Autopilot") onto a
modern JetPack 6 / L4T R36 stack, with the pipeline rebuilt in Python and the
surveillance-detection logic made explicit and testable.

---


## What Scout did, and where it lives here

| Scout | Here | Notes |
|---|---|---|
| `checkCopy.sh` — pull clips off the USB image | `teslacam-usb.sh`, `scripts/usb_*.sh` | USB gadget mode; already worked, untouched |
| `preprocess.sh` — ffmpeg keyframes, filename → date/time/camera | `src/clipmeta.py` | Regex parse instead of `cut`/`rev`, which mis-parsed modern camera names |
| YOLOv3 plate weights + ALPR-Unconstrained (WPOD-NET) | `src/alpr.py` | YOLO11n plate detector → geometric rectification → Tesseract |
| FaceNet (TF 1.x) + `facenet_trt` | `src/faces.py` | YuNet + SFace via OpenCV; same 128-d embeddings, builds on JetPack 6 |
| MongoDB: `polls`, `drives`, `geocodes`, `plates`, `plateDetections`, `faces`, `faceDetections` | `src/db.py` | Same model, SQLite |
| `scripts/TeslaJS/poll.js` — Owner API telemetry | `src/poller.py` | Fleet API (Owner API is retired), and it won't wake a sleeping car |
| Express + Vue: Recent / AllPlates / AllFaces / Timeline / Settings | `src/ui_app.py` | Same views, server-rendered; no node_modules, no second container |
| *(dedupe only — judgement left to the operator)* | `src/correlate.py` | **New.** Scores entities and explains why |
| `makeStranger` (split a face out) | `src/faces.py` + entity page | Plus the merge direction Scout lacked |
| *(never shipped)* | `src/notify.py` | **New.** ntfy / webhook / Pushover push |

### Not ported

- **MongoDB.** Every query Scout ran was a filter-and-sort; SQLite does that
  with real indexes and no daemon eating RAM on a 7.4 GB box.
- **The Vue client.** It was a read-mostly list-and-map UI. Server-rendered
  HTML gives the same thing with no build step and loads faster over a hotspot.
- **Scout's YOLOv3 plate weights.** The Google Drive link in its README (step
  24) is long dead. `scripts/fetch_models.sh` pulls a maintained YOLO11n plate
  model instead.

---

## The part that matters: surveillance detection

Scout deduplicated detections (same plate, more than 60 seconds apart) and drew
them on a timeline for a human to interpret. That dedupe is the right primitive
and is kept verbatim — but the judgement is now made explicitly, because the
signal isn't *how often* you saw a plate, it's *how those sightings are spread
across space, time and separate journeys*.

```
A plate seen 40 times in your driveway         → your neighbour       (0 points)
A plate on 3 drives across 2 days, 4 locations → follows you around   (alarming)
A plate 4 times in one drive, 6 miles apart,
  always in the rear cameras                   → behind you right now (alarming)
```

Five bounded signals combine into a 0–100 score: separate **drives**, separate
**days**, distinct **locations**, geographic **spread**, and **rear-camera**
bias. Then two suppressors, which matter more than the signals:

- **Anchored** — if every sighting clusters to one place, the score is
  multiplied by 0.15. A stationary object cannot follow you. This is applied
  only when GPS is actually available, because "always in one spot" and "we
  don't know where" are different things.
- **Dwell** — an entity seen only where *you* habitually park (learned from
  your own parked telemetry) is multiplied by 0.35. That's your home, office
  and gym, where innocent vehicles accumulate sightings simply because you're
  there a lot.

Plus the whitelist: marking an entity **known** in the UI removes it from
scoring permanently. It's the highest-leverage action in the system.

Separately, `detect_active_tail` looks for the real-time case — ≥3 encounters
within a *single* drive, spread over ≥2 miles and ≥5 minutes. Something that
satisfies all three didn't merely share a road with you.

`tests/test_correlate.py` encodes these as scenarios (neighbour, traffic,
multi-drive follower, active tail, whitelisted, no-GPS). If a weight change
makes the neighbour look like a stalker, the suite fails.

---

## Setup

```bash
git clone <this repo> && cd tesla-alerts
cp .env.example .env          # set DASHBOARD_USER / DASHBOARD_PASS at minimum
./scripts/fetch_models.sh     # YuNet, SFace, YOLO plate detector (~43 MB)
docker compose up -d --build
```

Dashboard: `http://<jetson>:8080`

### Optional: Tesla telemetry

Without credentials everything runs, but detections carry no GPS, so the
geographic signals score zero and the engine leans on drives/days/cameras
alone. With them, put a Fleet API access + refresh token in `.env` and the
`poller` service builds `polls`, `drives` and `geocodes`.

### Optional: notifications

Set any of `NTFY_TOPIC`, `ALERT_WEBHOOK_URL`, or `PUSHOVER_TOKEN` +
`PUSHOVER_USER`. Only `high` findings push by default (`NOTIFY_MIN_SEVERITY`).

### Backfill existing footage

```bash
docker compose exec processor python -u /app/scripts/backfill.py
```

Re-runs the new stages over clips already in `processed/`. Resumable.

---

## Architecture

```
Tesla ──USB-C──> Jetson gadget (3550000.usb) ──> /mnt/teslacam ──┐
                                                                 │
                          NATIX VX360 <── natix_worker (USB-A) ──┤
                                                                 │
  processor ──┬─ ingest (stability-checked copy)
              ├─ clipmeta   camera + capture time from filename
              ├─ YOLO       people and vehicles
              ├─ alpr       localize → rectify → legibility gate → OCR → vote
              ├─ faces      YuNet detect → SFace embed → cluster to identity
              └─ correlate ─> notify

  poller    ── Tesla Fleet API ─> polls / drives / geocodes
  gif_worker ─ ffmpeg ─> 5s GIF per alert
  web       ── FastAPI dashboard + JSON API
                       ↑
                  scout.db (SQLite, WAL)
```

Services: `processor`, `gif_worker`, `poller`, `web`. All four share
`/mnt/jetsondata/tesla-alerts` as `/data`. A fifth, `natix-mirror`, runs on the
host rather than in Docker - see below for why.

### Two design decisions worth knowing

**Plate localization runs on vehicle crops, not the whole frame.** Ultralytics
letterboxes whatever you hand it down to `imgsz` (640). A front-camera frame is
2896×1876, so passing the whole frame scales plates down 4.5× before the model
sees them. Raising `imgsz` is the wrong fix — this checkpoint was *trained* at
640, and inferring at 1920 mostly produced confident false positives (the most
convincing "plate" it recovered was a Chipotle storefront sign). Instead we crop
to the vehicles YOLO already found and run the detector on each at native
resolution, unioned with a whole-frame pass and NMS-deduped. Measured on real
footage: the same plate goes from 75×47 @ conf 0.42 to 102×69 @ conf 0.79.

**Frames are spread across the whole clip.** The obvious sampling — N fps for the
first 10 seconds — looks at one sixth of a 60-second clip and measurably loses
detections; this dataset's clearest plate is 26 seconds in. The same 30-frame
budget spread over the full minute costs identical inference time and gives 6×
the temporal coverage. For "was this vehicle present during this minute", that
trade is strongly correct.

**Cross-frame plate voting.** Single-frame plate OCR from a moving dashcam is a
coin flip. Every read from a clip is grouped by confusable-glyph equivalence
(B/8, O/0, I/1 — the errors OCR always makes) and voted, weighted toward frame
support. A mediocre read seen ten times beats one lucky read of something else.
Scout read each frame independently.

**The legibility gate.** Plate crops from a dark car park are real plates that
happen to be unreadable, and Tesseract asked to read one will still return
*something*. Those characters are noise, and noise is expensive here: it mints a
fake vehicle identity the correlation engine then reasons about. So crops below
a brightness/contrast floor are rejected before OCR, and counted separately —
"no plates" and "plates too dark to read" are different problems.

---


## Feeding the NATIX VX360

### The problem this solves

The [NATIX VX360](https://www.natix.network/vx360-depin-dashcam) is not a
camera. It is a USB stick with an ARM Cortex-A7 running Armbian inside it. You
normally plug it into the Tesla's glovebox port *instead of* a dashcam drive:
the car writes TeslaCam footage onto it, and when the car is parked on a known
WiFi network the stick uploads that footage to NATIX's cloud by itself.

Here, the car's USB line goes to the **Jetson**, because that is the only way
the Scout pipeline gets to see every frame. The direct consequence is that the
VX360 sees nothing at all - the car is no longer talking to it.

So the Jetson takes over the writing. Both devices connect at once, on
different ports, because the Jetson has both roles available:

| Port | Mode | Talks to |
|---|---|---|
| USB-C (`3550000.usb`) | **device** - the only UDC on the board | the car, as a mass-storage gadget |
| USB-A ×4 | **host** | the VX360, as an ordinary USB stick |

There is no contention between them. The car sees a dashcam drive; the Jetson
sees a USB stick; neither knows about the other.

### What the mirror does

`src/natix.py` copies every clip in the archive onto the stick, into the exact
layout the car would have produced:

```
/mnt/natixv360/TeslaCam/SentryClips/2026-02-16_20-49-20/
    2026-02-16_20-49-20-front.mp4
    2026-02-16_20-49-20-back.mp4
    2026-02-16_20-49-20-left_repeater.mp4
    ...
    event.json          ← synthesised if the car didn't write one
```

The stick's firmware is looking for precisely that and has no idea a Jetson is
standing in for a Tesla.

Four properties are load-bearing:

- **Crash safety.** Every file is written to a temp name on the destination and
  `os.replace`d into position after an `fsync`. The car cuts power without
  warning; a truncated clip must never appear under a real filename.
- **Idempotence.** If the rename landed but the database write didn't, the next
  pass notices the file is already there and the right size, records it, and
  moves on rather than recopying 40 MB. A file that is present but *short* is
  recopied.
- **Loop recording.** When the stick fills, our oldest mirrored *events* are
  pruned whole. Pruning is driven entirely by the `natix_mirror` table, so
  anything we did not write - the stick's own metadata, the user's files -
  survives.
- **Refusal.** Discovery reports every candidate with a confidence and its
  reasons, and writes only at or above `NATIX_MIN_CONFIDENCE`. The boot card,
  the Scout data volume, a read-only volume, and whatever the USB gadget is
  currently exporting to the car are all excluded outright.

### Identifying the stick

Run the probe with the stick plugged into any USB-A port:

```bash
python3 scripts/natix_probe.py          # judge what is attached
python3 scripts/natix_probe.py --tree   # mount read-only and list contents
```

On the unit in this car it reports:

```
/dev/sda1   PINNED   USABLE
     size        37.6 GB          ← the exposed LUN, not the 128-256GB of flash
     filesystem  exfat
     label       VX360
     uuid        FFFB-6B2F
     vendor      Linux            ← the stick's own kernel, not a vendor name
     model       File-Stor Gadget
     usb id      0525:a4a5
```

**Pin it on the volume UUID, not the serial or the USB id.** The VX360 exports
its flash with the Linux kernel's file-backed storage gadget, so it reports
that gadget's stock identifiers - `0525:a4a5` and
`Linux_File-Stor_Gadget-0:0`. Every Linux storage gadget reports the same ones,
*including this Jetson* when it is presenting a drive to the car. `natix.py`
knows this and refuses to treat either as an identity even if you set them; the
volume UUID is the thing that is actually unique.

Two other findings worth knowing about the hardware: the stick exposes only
about **35 GiB** of its advertised 128-256 GB (it keeps the rest for its own OS
and upload spool), and it presents **exFAT**, which this L4T 5.15 kernel has no
driver for - so mounting goes through `exfat-fuse`, which is why
`scripts/natix_mount.sh` special-cases it.

### Installing it

```bash
sudo ./scripts/install_natix.sh
```

Root is needed for exactly one thing - mounting a block device. `/dev/sda1` is
`root:disk 0660` with no ACL for the login user, and this is a headless box, so
the usual unprivileged routes are closed: `udisksctl` fails because polkit
treats a seatless SSH session as inactive, and FUSE cannot help because
`exfat-fuse` still needs to open the device.

The script does the whole job in that one privileged run, in this order:

1. installs the unit and the udev rule
2. **identifies the stick** - aborts if it cannot
3. **runs the first full mirror in the foreground**, with progress
4. **verifies independently** what actually landed, including sha256 on a sample
5. **only then** enables the service for continuous mirroring

Steps 2-4 are the point. A background service that fails to mount looks exactly
like a background service with nothing to do, and you would not find out until
you went looking for footage that was never uploaded. Any failure aborts
without enabling the service, so there is no half-installed state quietly
retrying a broken configuration forever.

```bash
journalctl -u natix-mirror -f                                  # watch it work
sudo BASE_DIR=/mnt/jetsondata/tesla-alerts \
     python3 src/natix_worker.py --status                      # one-shot report
sudo BASE_DIR=/mnt/jetsondata/tesla-alerts \
     python3 scripts/natix_verify.py --sample 10               # re-verify the stick
sudo ./scripts/install_natix.sh --uninstall                    # undo everything
```

`natix_verify.py` reads back from the stick rather than trusting the mirror's
own log - "copied 132 clips" and "132 playable clips are on the stick" are
different claims, and only the second one matters when NATIX's cloud is going
to upload them. It compares sizes against the *source files*, not against the
sizes the mirror recorded, and hashes a random sample end to end.

The dashboard's **NATIX** tab shows the same state, read out of the database.

### The stick arrives full, and will never empty itself

The first real install hit this immediately: **362 MB free of 37.6 GB**. Not a
misconfiguration - a structural consequence of the topology.

A VX360 loop-records while the *car* writes to it. Here the car writes to the
Jetson, so the stick's own loop-deletion never runs again. It stays frozen at
whatever state the car left it in, permanently full. Meanwhile the mirror's own
pruning only removes footage *we* wrote, of which there is none on a fresh
stick. Deadlock: nothing we send will ever fit.

So in this arrangement **the loop has to become our job**. That is what
`NATIX_RECLAIM_BUCKETS` is for, and it is empty by default because the footage
belongs to the owner and only they know whether it has been uploaded yet.

```bash
# look first - read-only, modifies nothing, shows per-bucket GB and event ages
sudo python3 scripts/natix_probe.py --tree

# then, if the space is old car footage you are happy to loop over:
NATIX_RECLAIM_BUCKETS=SentryClips,RecentClips
```

Reclaiming is fenced in hard, because it is the one place this system destroys
data it did not create. An entry is eligible only if **all** of these hold:

- it is under `TeslaCam/<bucket>/`, and `<bucket>` is one of Tesla's three real
  bucket names **and** was named explicitly in `NATIX_RECLAIM_BUCKETS`
- its name parses as a Tesla event stamp or clip name - a stray directory is
  skipped, not deleted
- it is not an event the mirror itself wrote (`prune_oldest` owns those; double
  ownership is how files get deleted twice)
- it is not a symlink and not a dotfile

It deletes oldest-recording-first *across* buckets, stops the moment there is
enough room rather than wiping, and logs every entry by name and size to the
journal. `SavedClips` should stay out of any value you set - those are the clips
somebody deliberately pressed the button to keep. Eleven tests in
`tests/test_natix.py` cover exactly these refusals.

### What the real stick turned out to hold

Measured on the unit in this car, and worth recording because none of it was
guessable:

| | files | bytes |
|---|---|---|
| `.Trashes`, `.Spotlight-V100`, `.fseventsd` (macOS) | 192 | 3.48 GB |
| `TeslaCam/EncryptedClips` (NATIX's own upload spool) | ~69 | 2.74 GB |
| **every file on the volume** | **261** | **6.22 GB** |
| what `df` calls used | | **34.65 GB** |

`TeslaCam/SentryClips` was **empty** - there is no old dashcam footage to loop
over, so `NATIX_RECLAIM_BUCKETS` correctly finds nothing eligible.

The file count reconciles exactly with `fsck.exfat`'s own (`clean. directories
35, files 261`), so nothing is hidden from the walk. That leaves ~28.4 GB of
clusters marked used in the allocation bitmap while no directory entry
references them - leaked by a device that loses power mid-write every time the
car sleeps, and has also been pulled out of a Mac. `fsck.exfat -n` reports
`clean` regardless, so exfatprogs 1.1.3 evidently does not cross-check the
bitmap in no-modify mode.

**How much of that is measured, and how much is inference.** For a long time
it was mostly inference, and the report presented it as fact. Both the Python
walk and the "independent" `du` cross-check totalled *apparent* size - `du -sb`
implies `--apparent-size` - so they measured the same dimension twice and the
agreement between them proved nothing about the third. What was never measured
was *allocated* size: what the files occupy rather than what they contain. Two
completely different faults produce the same shortfall against `df`, and they
have opposite remedies:

| what the numbers show | fault | remedy |
|---|---|---|
| allocated ≈ `df` used, apparent much lower | files occupy space they do not contain | delete those files |
| both far below `df` used | clusters belong to no file | `--repair` or reformat |

Both tools now report both dimensions and name which fault they are looking
at, and on 2026-09-01 the measurement came back:

```
4.45GB  <- total the files contain
4.46GB  <- total the files occupy
```

Both fall ~28.5 GB short of what `statvfs` calls used, so it is **confirmed
leaked clusters** rather than inferred. The ~10 MB between the two totals is
cluster rounding (116 files x 128 KB cluster = 14.5 MB maximum), and its being
non-zero is itself the evidence that the FUSE driver reports real allocation
rather than echoing `st_size` - so the degraded-check warning correctly stayed
quiet and the numbers can be believed.

Note that `fsck.exfat -n` still called the volume `clean` in the same run. That
is not a contradiction to resolve in fsck's favour: exfatprogs 1.1.3 in
no-modify mode does not cross-check the allocation bitmap, so its verdict is
silent on precisely this fault. The script now says so at the point of
decision, because it previously printed the opposite - "if fsck said 'clean',
the space is genuinely occupied by real files and there is nothing to repair" -
directly beneath a walk that had just proved otherwise.

The reasoning that predicted this outcome: 
exFAT has no preallocation beyond the file size the way ext4 `fallocate` does -
the stream-extension entry's `DataLength` *is* `st_size`, so slack is bounded
by (files × cluster size), tens of megabytes here rather than tens of
gigabytes. The point is that this is now the conclusion the measurement
supports, rather than the one the report assumed.

One caveat the tooling states out loud: exFAT here is FUSE, not a kernel
driver, and a FUSE driver may derive `st_blocks` from `st_size` rather than
from the real cluster chain. If it does, the two totals agree by construction
and their agreement is not evidence. `natix_fsck.sh` detects that case
(identical totals across many files) and says the run cannot tell the two
faults apart, instead of letting a degraded check read as a passing one.

Two consequences:

- **`.Trashes` is the only recoverable space today** (3.48 GB via
  `--empty-trash`), which with the reserve intact is enough to mirror ~44 of
  132 clips. Partial, but the mirror is designed for exactly that - it copies
  newest-first, so a partial stick holds a contiguous stretch of the most
  recent history rather than an arbitrary scattering of it.
- **`--repair` does not recover it.** Tried on 2026-09-01: `fsck.exfat -y`
  reports `clean. directories 15, files 116` and changes nothing, exactly as
  `-n` did. exfatprogs 1.1.3 does not cross-check the allocation bitmap in
  either mode, so it has nothing to find and therefore nothing to fix. This is
  the limit of that tool version, not a property of the stick.
- **A reformat is the remaining remedy**, via `scripts/natix_reformat.sh`,
  which rescues and sha256-verifies every foreign file *before* it will format
  anything, then restores and re-verifies. Run it with no arguments first for a
  dry-run inventory.

There is one more trap in reformatting by hand: `mkfs.exfat` 1.1.3 has no
option to set the volume UUID, so a format mints a new random one and the
`NATIX_VOLUME_UUID` pin in `.env` silently stops matching. Identity then
quietly degrades to matching on the volume label alone. `natix_reformat.sh`
reads the new UUID and rewrites the pin, keeping a timestamped `.env` backup.

**Do not reformat this stick without first copying `TeslaCam/EncryptedClips`
off it.** That is not general caution, it is specific: the spool holds clips
dated **2026-08-03**, and the Jetson archive holds **132 clips spanning
2026-02-16 12:49-13:42 and nothing else at all**. Zero clips from August. So
the 2.74 GB spool is the only copy of that footage in existence, and a reformat
destroys it permanently. The clips this mirror wrote are the opposite case -
reproducible from the archive at any time, and free to lose.

Two things follow from that date range which are worth chasing separately:
the stick has not uploaded in four weeks, which points at it having no WiFi;
and the Jetson's own archive has ingested nothing since February.

Do NOT set `NATIX_IGNORE_FREE_SPACE` for this. The free-space figure was
verified accurate by `--write-test`: 362 MB free, 256 MB written, 106 MB left.

A note on the reserve. It is tempting to lower `NATIX_RESERVE_MB` to fit more
clips, and it is the wrong move: the stick has to *re-encrypt* each clip into
`EncryptedClips` before uploading it. Fill the volume and its uploader has
nowhere to work, so more mirrored clips would mean fewer uploaded ones.

### Why the unit has almost no systemd sandboxing

`deploy/natix-mirror.service` deliberately sets **none** of `ProtectSystem`,
`ProtectHome`, `ReadWritePaths`, `PrivateTmp`, `ProtectKernelTunables` or
`SystemCallFilter`. That is not laziness, and it cost real data to learn.

Any one of those puts the unit in a private mount namespace. `ReadWritePaths`
goes further: systemd **bind-mounts each listed path onto itself** so it can be
made writable. Listing `/mnt/natixv360` there meant systemd had already mounted
something at the mountpoint before the service body ran:

```
mount failed: /mnt/natixv360 already has /dev/mmcblk0p1[/mnt/natixv360] mounted, not /dev/sda1
```

Before the mount verification existed, that is the most likely reason a mount
"succeeded" while nothing was attached - and 1.1 GB of clips went onto the
Jetson's own SD card instead of the stick. Writing to a directory always works,
so nothing complained until FUSE started refusing with `mountpoint is not
empty`.

**A service whose entire job is mounting a filesystem at a fixed path cannot be
sandboxed by rearranging its mounts underneath it.** What remains are the
protections that do not touch the mount table: `NoNewPrivileges`,
`RestrictRealtime`, `RestrictSUIDSGID`, `LockPersonality`, `MemoryMax`.

`SystemCallFilter=@system-service` is a particular trap: that set *excludes*
`@mount`, so it would block the syscall this service exists to make.

### Verified working

```
[PASS] every recorded clip exists on the stick          40/40 present
[PASS] sizes match the source archive                   40/40 exact
[PASS] sampled clips are byte-identical to the source   8/8 sha256 match
[PASS] paths parse as TeslaCam/<bucket>/<event>/...     40/40 well-formed
[PASS] every event folder has an event.json             7/7 folders
[PASS] no partial (.part) files remain
[PASS] the mirrored window is the newest end            40/132, rolling window
[PASS] no clip copied and then evicted (churn)          0 pruned rows
[PASS] stick retains its reserve of free space          2.06 GB free
```

Three bugs only the real device could produce, all fixed and regression-tested:

1. **The rewrite loop.** `build_plan` excluded `state='done'` but not
   `'pruned'`, so an evicted clip looked un-mirrored and was recopied, evicting
   something else - 60 done / 72 pruned oscillating forever at 9 MB/s. Fixed by
   excluding pruned, mirroring newest-first, and refusing to evict anything
   newer than the candidate.
2. **Writing to the host disk.** Covered above.
3. **The verification demanding the impossible.** It asserted all 132 clips
   were mirrored onto a stick that holds 40. The right invariant for a rolling
   window is contiguity: everything dropped must be older than everything kept.

### Live status and logs over the tunnel

`https://azula.tedebyte.dev/natix` shows the mirror's state and the tail of the
host service's log, refreshing every 5 seconds.

```bash
sudo ./scripts/expose_natix_tunnel.sh --dry-run   # show the ingress diff
sudo ./scripts/expose_natix_tunnel.sh             # apply, validate, reload
sudo ./scripts/expose_natix_tunnel.sh --remove    # take it back out
```

Two details that are easy to get wrong:

**Rule order.** cloudflared matches ingress rules top-down, and
`azula.tedebyte.dev` already has a catch-all sending every path to the
spv-server on `:7878`. A rule appended at the end would never be reached, so
the script inserts before that catch-all and refuses to guess if the config
does not have the shape it expects. It validates with
`cloudflared tunnel ingress validate` *before* replacing the live file - a
tunnel that will not start takes down every hostname on it, not just this one -
and keeps a timestamped backup.

**Scope.** Only `^/natix|^/api/natix` is published. The rest of the dashboard
(Recent, Plates, Faces, Timeline, Settings, `/media`) stays reachable only over
the tailnet. Widen the regex if you want the whole thing. Both layers of auth
apply: Cloudflare Access fronts the hostname, and the dashboard adds HTTP Basic
from `DASHBOARD_USER` / `DASHBOARD_PASS`.

**Why the log goes to a file at all.** The mirror is a systemd unit on the host
and the dashboard is a container, so journald is out of reach and plumbing it
in would mean new mounts and privileges for a read-only convenience. Instead
the worker writes the same lines it prints to `BASE_DIR/logs/natix.log`, which
the container already has mounted as `/data`. It rotates at 2 MB, one
generation deep - this is a car appliance sharing a 119 GB card with the clip
archive, and nobody is going to read `natix.log.7`.

The page polls rather than streaming: a long-lived SSE response through a
Cloudflare tunnel is the thing most likely to be buffered or cut, and two small
JSON fetches every 5 seconds are boring and reliable.

### The one thing that cannot be verified from here

Whether the VX360's firmware notices files that appeared while it was exporting
its flash to us, or whether it only rescans after a physical replug. We flush
and unmount after every pass, which is the strongest hint we can send. Confirm
in the NATIX app that the mirrored clips queue for upload; if they do not,
unplug and replug the stick once and check again - that result distinguishes
"needs a detach" from "needs something else entirely".

### Why this one service is not a container

Every other service here is a container. This one is a systemd unit running as
root, for two concrete reasons:

1. A mount made inside a container lands in that container's mount namespace.
   Bind-propagating `/mnt` as `rshared` into a privileged container would work,
   but it trades a clear ownership boundary for a subtle one on a machine that
   loses power without warning.
2. `scout.db` is written by root-owned containers, so a non-root writer is
   locked out of the very database it has to record copies in.

The considered alternative - an unprivileged service plus a `NOPASSWD` sudoers
rule for `natix_mount.sh` - creates a permanent escalation path for a service
that needs root for half its job anyway. One root service is the smaller
surface. It is confined in the unit (`ProtectHome=read-only`,
`ReadWritePaths=`, `NoNewPrivileges=`, `MemoryMax=512M`), and
`natix_mount.sh` validates independently: USB transport only, an allowlisted
mountpoint only, never the LUN currently exported to the car, and always
`nosuid,nodev,noexec`.

## Where the car is: GPS, /findmy and /gps

Location comes from a **USB u-blox receiver** on the Jetson rather than Tesla's
Fleet API. The Fleet API still works and the two write the same `polls` rows, but
the receiver needs no credentials, cannot be deprecated out from under the project
the way Scout's `owner-api.teslamotors.com` was, does not keep a parked car awake,
and works with no connectivity at all - which matters, because this runs in a car.

What it gives up is odometer, charge state and shift position. Nothing in
`correlate.py` scores on any of those.

### The hardware

`1546:01a8 u-blox 8`, on a USB extension. The chipset is not a detail: this
kernel builds `cdc-acm`, `cp210x` and `ftdi_sio` but **not** `pl2303` or `ch341`,
so the popular GlobalSat BU-353 pucks (Prolific) enumerate and then produce no
device node at all. Always address the receiver by its stable path -
`/dev/serial/by-id/usb-u-blox_AG_-_...` - never `/dev/ttyACM0`, which is
assigned in enumeration order and moves when anything else CDC-ACM is plugged in.

### Two pages

| Page | Question it answers |
|---|---|
| `/findmy` | Where is the car, how old is that answer, and is anything still watching it |
| `/gps` | Why is there, or is there not, a fix - one layer at a time |

Both **fully answer their question with JavaScript disabled**. The coordinates,
address, absolute time and staleness band are server-rendered; the map is an
enhancement layered on top. A car with no signal is the normal case here, so a
map that cannot load must never remove the answer.

### The two things this design is really about

**A parked car and a dead service are byte-identical in `polls`.** `gps.py`
skips the database write whenever the fix is invalid, so a healthy receiver in an
underground garage writes exactly nothing - the same as a crashed process, an
unplugged receiver, or a service that was never installed. A page that reads only
`polls` states a coin flip as a fact, precisely when it is most needed. That is
why `gps.py` publishes a heartbeat (`logs/gps.json`, atomic tmp+fsync+rename) and
why `/findmy` joins the two before saying anything. It is also why the heartbeat
is written from inside `read_sentences`' exception handlers: an unplugged
receiver yields no sentences at all, so a heartbeat written only on the success
path freezes, and "receiver unplugged" renders as "service dead".

**Satellites in view prove nothing about the antenna.** "In view" comes from the
receiver's stored almanac and needs no reception whatsoever - a receiver with a
disconnected antenna still reports satellites in view for hours. Only a nonzero
C/N0 proves the RF path works. `/gps` shows in-view, tracked and used separately
for exactly this reason, and the highest-value indoor reading is *in view, zero
tracked*: the almanac knows where to look and the receiver hears nothing.

### Testing it indoors

A receiver streams NMEA whether or not it has a fix, so the whole chain is
testable with no sky:

    sudo usermod -aG dialout tedster        # then log out and back in
    python3 src/gps.py --status             # touches no database, safe alongside the service
    python3 src/gps.py --raw                # the sentences themselves

Sentences arriving with empty position fields (`$GNRMC,,V,,,...`) prove USB,
driver, node and framing all work. Only the fix itself needs sky.

### Logs deliberately carry no location

`logs/gps.log` is served to the dashboard and sits on a volume four containers
mount. It logs status fields only - satellite counts, HDOP, fix quality, errors -
and never coordinates *or the geocoded street*, because once the car is parked
the street name **is** the home address. An earlier version scrubbed the numbers
and passed "1425 Camino De Los Coches" through untouched, which is worse than no
redaction: the redaction that is there makes the file look safe.


## Verifying the pipeline actually works

A zero result and a silently broken pipeline look **identical** from the dashboard.
Reading the code cannot tell them apart, and neither can a claim from whoever last
touched it. Three layers exist so you never have to take that on trust.

### 1. The positive control

```bash
docker compose exec processor python -u /app/scripts/selfcheck.py
```

Manufactures six clips across three days and three locations, each containing a
plate reading `7ABC123` and a detectable face, seeds the poll/drive tables, and
runs them through the **real** production functions — `register_clip`,
`analyze_clip`, `record_plate_detections`, `record_face_detections`,
`run_correlation_pass`. Then it asserts the known answer came back, stage by stage.

It runs entirely in a scratch directory and refuses to start if `BASE_DIR` or
`DB_PATH` resolve inside `/data`, so it cannot touch your real footage.

If this passes, a zero result on real footage is a statement about **the footage**.
If it fails, it names the stage.

```
[PASS] alpr       plate text matches expected   read ['7ABC123'], expected 7ABC123
[PASS] faces      identities stable across clips 4 after clip 1, 4 at the end
[PASS] correlate  plate scored as a high-severity finding  score 100.0 severity 'high'
RESULT: PASS - all 20 checks passed.
```

It also prints what it does **not** cover — ingest, GIF/notification delivery, and
whether the thresholds suit dark real-world footage. Read that list; it is the
honest boundary of the guarantee.

### 2. Proof the positive control is not vacuous

```bash
docker compose exec processor python -u /app/scripts/mutation_test.py
```

A control that passes unconditionally is *worse* than none — it converts "we
haven't checked" into "we checked and it's fine". So this deliberately breaks the
pipeline ten different ways (plate reader returns nothing; every crop judged
illegible; OCR returns empty; detections never persisted; face detector blind;
embeddings dropped; identities never written; encounters never collapse; every
score zeroed; the analyser returns nothing) and confirms the control catches each
one and names the right stage.

It copies the repo per mutation, so your working tree is never modified, and runs
an unmutated **baseline** first — if that does not pass, it aborts rather than
report meaningless results.

```
mutation               caught   right stage
plate-blind            yes      yes
...
RESULT: PASS - all 10 mutations were caught.
```

### 3. Standing behaviour tests

```bash
docker compose exec processor sh -c 'for t in /app/tests/test_*.py; do python -u "$t"; done'
```

24 scenarios pinning the judgement calls: the neighbour scores low, the
multi-drive follower scores high, a whitelisted entity scores zero, encounter
merging is order-independent, OCR survives confusable glyphs.

### When to run what

| Situation | Run |
|---|---|
| Changed anything in `src/` | tests, then `selfcheck.py` |
| Changed a threshold or a gate | `selfcheck.py` — it exercises every gate |
| Changed `selfcheck.py` itself | `mutation_test.py` |
| A real run produced zero and you doubt it | `selfcheck.py` — that is exactly its job |
| Before deploying to the car | all three, plus `./scripts/fetch_models.sh` |

## Tests

```bash
python3 tests/test_correlate.py   # scoring scenarios (no deps)
python3 tests/test_alpr.py        # OCR + voting (needs cv2; skips without tesseract)
python3 tests/test_faces.py       # identity clustering (needs cv2, numpy)
python3 tests/test_natix.py       # VX360 mirror: refusal, crash safety, pruning (no deps)
```

Or inside the container, where every dependency is present:

```bash
docker compose exec processor sh -c 'for t in /app/tests/test_*.py; do python -u $t; done'
```

---

## Known limits

- **Night plate reads are the binding constraint.** On the sample footage here
  (a single rainy 53-minute night session, every clip's mean frame luminance
  below 47), plates localize at 76–110 px wide and Tesseract returns *zero* text
  rows — not low-confidence rows, none. The legibility gate rejects them rather
  than inventing characters. Verified four independent ways, including forcing
  OCR past the gate. Daylight and closer following distances read far better;
  this is a physics problem, not a code one.

- **Scoring needs more than one session.** Four of the five signals (drives,
  days, locations, spread) require GPS and multiple journeys. On a single-day,
  no-GPS dataset the reachable maximum score is exactly 35.0 — which is exactly
  the medium threshold — and high is unreachable. This was established by
  enumerating the *entire* configuration space, not by sampling. Connect the
  Tesla API and drive for a few days before judging the engine.
- **Inference is CPU-bound, and that caps throughput.** The container is
  `python:3.10-slim`, which has no CUDA wheels for aarch64, so everything runs
  on the Orin's CPU. Measured on this box, per 10-second clip at the default
  3 fps sampling:

  | stage | busy clip (168 vehicles) | quiet clip |
  |---|---|---|
  | decode + sample (30 frames) | 2.2 s | 2.4 s |
  | main YOLO | 27.7 s | 27.4 s |
  | + ALPR | +15.3 s | +0.4 s |
  | + faces | +0.0 s | +0.4 s |
  | **full pipeline** | **40.7 s** | **29.8 s** |

  The main YOLO pass is two thirds of the cost, and it scales linearly with
  `SAMPLE_FPS`. ALPR and faces are cheap when there is nothing to look at,
  because both are gated on YOLO having found a vehicle or a person first.

  A Tesla records one-minute clips from four to six cameras at once, so live
  ingest produces 4–6 clips per minute while we process 1.5–2. **Set
  `SAMPLE_FPS=1.0` for roughly real-time throughput** (~9–14 s/clip), or move
  to an `l4t-pytorch` base image to put YOLO on the GPU and keep the sampling
  rate. The default of 3.0 is right for backfilling and for sentry-only use.
- **Face identities drift.** Online clustering with a running mean is O(people)
  rather than O(sightings), which is what keeps it real time, but it can split
  one person across two identities under very different lighting, or merge two
  people on a bad frame. Both are repairable from an identity's page — **Merge
  this into…** combines two strangers, and **not this person** on a single
  sighting pulls it out into a new one (Scout's `makeStranger`). Both recompute
  the affected identities' embeddings from their remaining sightings, so future
  matching stays correct.
- **No GPS means weaker scoring**, as described above.

---

## Legal

Same caveat Scout shipped with, and it is not boilerplate. This processes video
of people and vehicles in public and builds a searchable history of where they
were seen. Recording, retention and biometric-processing law varies enormously
by jurisdiction — Illinois BIPA, Texas CUBI, GDPR and the UK DPA all bear on
running face recognition over footage of people who haven't consented, even on
your own vehicle. It is your responsibility to know what applies to you.

Provided for educational and personal-security use, at your own risk.
