"""
Shared "one card = one item" upload-slot deck component.

Used by BOTH app.py's "Upload Your Items" step and
description_generator.py's item-ingestion step — pulled into its own
module (rather than duplicated) specifically so the two stay visually
and behaviorally IDENTICAL by construction, not by two hand-maintained
copies drifting apart over time.

Each card shows a large main photo (the first path in the slot) + up
to 6 thumbnails, or a blank-state dropzone + "+" placeholders before
any photos exist. Click-to-browse and drag-reorder reuse the exact
prepareFile()/pointer+FLIP mechanics already proven in app.py's own
item-editor component (ITEM_CARD_JS) — adapted here, not reinvented.

Callers own their own upload directory and their own slot list shape
(each slot: {"slot": i, "paths": [...], "vintage": bool, ...}) — this
module only touches "paths"/"vintage" on whichever slot dict it's
given, via handle_upload_slot_action().
"""

import base64
from pathlib import Path

import streamlit as st

try:
    from streamlit.components.v2 import component as _streamlit_v2_component
    STREAMLIT_V2_AVAILABLE = True
except ImportError:
    _streamlit_v2_component = None
    STREAMLIT_V2_AVAILABLE = False


UPLOAD_SLOT_DECK_HTML = """
<div class="slot-deck" data-deck></div>
"""

UPLOAD_SLOT_DECK_CSS = """
.slot-deck {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 14px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
@media (max-width: 1180px) {
  .slot-deck { grid-template-columns: repeat(3, 1fr); }
}
@media (max-width: 720px) {
  .slot-deck { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 440px) {
  .slot-deck { grid-template-columns: 1fr; }
}

.slot-card {
  position: relative;
  background: #FFFFFF;
  border: 1px solid rgba(43, 33, 48, .14);
  border-radius: 12px;
  padding: 10px;
  box-shadow: 0 6px 14px rgba(0, 0, 0, .10);
  color: #2B2130;
  transition: border-color .12s ease, box-shadow .12s ease;
}
.slot-card.file-over {
  border-color: #F02AA0;
  box-shadow: 0 0 0 3px rgba(240, 42, 160, .18), 0 10px 24px rgba(240, 42, 160, .18);
}
@keyframes slotSpin { to { transform: rotate(360deg); } }

/* Per-photo pending state — a small corner chip while a just-selected
   local preview is being encoded/saved, so the photo itself stays
   fully visible instead of being hidden behind a full-card overlay. */
.slot-pending-overlay {
  position: absolute;
  top: 4px;
  right: 4px;
  background: rgba(255, 255, 255, .92);
  border-radius: 999px;
  padding: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 6px rgba(0, 0, 0, .15);
}
.slot-pending-overlay .spinner {
  width: 14px;
  height: 14px;
  border: 2.5px solid rgba(240, 42, 160, .25);
  border-top-color: #F02AA0;
  border-radius: 50%;
  animation: slotSpin .7s linear infinite;
}
.slot-pending-overlay.is-failed {
  position: absolute;
  inset: 0;
  top: auto;
  right: auto;
  border-radius: inherit;
  background: rgba(255, 255, 255, .94);
  flex-direction: column;
  gap: 4px;
  padding: 6px 20px 6px 6px;
}
.slot-pending-label { font-size: 9px; font-weight: 800; color: #e0335c; text-align: center; line-height: 1.2; }
.slot-main-photo .slot-pending-label { padding-right: 14px; }
.slot-retry-btn {
  border: none;
  background: #F02AA0;
  color: #fff;
  font-size: 9px;
  font-weight: 800;
  padding: 3px 10px;
  border-radius: 999px;
  cursor: pointer;
}
.slot-dismiss-btn {
  position: absolute;
  top: 3px;
  right: 3px;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  border: none;
  background: rgba(43, 33, 48, .7);
  color: #fff;
  font-size: 10px;
  line-height: 1;
  cursor: pointer;
}
.slot-main-photo.is-failed img, .slot-thumb.is-failed img { opacity: .35; }

.slot-header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 8px;
}
.slot-badge {
  font-size: 11px;
  font-weight: 800;
  letter-spacing: .02em;
  background: linear-gradient(135deg, #F02AA0, #FF6EC7);
  color: #fff;
  padding: 3px 8px;
  border-radius: 999px;
  flex-shrink: 0;
}
.slot-meta { font-size: 11px; color: #6B5B66; font-weight: 600; flex: 1; }
.slot-clear-btn {
  border: none;
  background: transparent;
  color: #6B5B66;
  font-size: 16px;
  cursor: pointer;
  line-height: 1;
  padding: 2px 5px;
  border-radius: 6px;
  flex-shrink: 0;
}
.slot-clear-btn:hover { background: rgba(43, 33, 48, .08); color: #2B2130; }

.slot-body { display: flex; gap: 8px; }

.slot-main-drop {
  flex: 0 0 38%;
  aspect-ratio: 2 / 3;
  max-height: 148px;
  min-height: 84px;
  border: 1.5px dashed rgba(240, 42, 160, .35);
  border-radius: 10px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
  text-align: center;
  cursor: pointer;
  background: linear-gradient(160deg, rgba(255, 193, 233, .16), rgba(251, 243, 227, .5));
  transition: border-color .12s ease, background .12s ease;
  padding: 6px;
}
.slot-main-drop:hover {
  border-color: #F02AA0;
  background: linear-gradient(160deg, rgba(255, 193, 233, .30), rgba(251, 243, 227, .7));
}
.slot-main-drop-icon { font-size: 16px; color: #F02AA0; }
.slot-main-drop-title { font-size: 10.5px; font-weight: 800; color: #2B2130; line-height: 1.2; }
.slot-main-drop-sub { font-size: 9px; color: #6B5B66; line-height: 1.2; }

.slot-main-photo {
  position: relative;
  flex: 0 0 38%;
  aspect-ratio: 2 / 3;
  max-height: 148px;
  min-height: 84px;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid rgba(43, 33, 48, .12);
  background: #f3ede6;
}
.slot-main-photo img { width: 100%; height: 100%; object-fit: cover; display: block; }
.slot-main-badge {
  position: absolute;
  left: 5px;
  bottom: 5px;
  background: rgba(43, 33, 48, .82);
  color: #fff;
  font-size: 8px;
  font-weight: 800;
  letter-spacing: .03em;
  padding: 2px 6px;
  border-radius: 5px;
}

.slot-thumb-grid {
  flex: 1;
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  grid-template-rows: repeat(3, 1fr);
  gap: 5px;
}
.slot-thumb {
  position: relative;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid rgba(43, 33, 48, .12);
  background: #f3ede6;
  cursor: pointer;
  touch-action: none;
  min-height: 26px;
}
.slot-thumb img { width: 100%; height: 100%; object-fit: cover; display: block; pointer-events: none; }
.slot-thumb.pointer-dragging { z-index: 9999; }
.slot-thumb.reorder-placeholder {
  background: rgba(240, 42, 160, .10);
  border: 1.5px dashed rgba(240, 42, 160, .4);
}
.slot-thumb-remove {
  position: absolute;
  top: 2px;
  right: 2px;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  border: none;
  background: rgba(43, 33, 48, .72);
  color: #fff;
  font-size: 11px;
  line-height: 1;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: opacity .12s ease;
}
.slot-thumb:hover .slot-thumb-remove { opacity: 1; }
.slot-thumb-placeholder {
  border-radius: 8px;
  border: 1.5px dashed rgba(43, 33, 48, .18);
  display: flex;
  align-items: center;
  justify-content: center;
  color: rgba(43, 33, 48, .3);
  font-size: 14px;
  font-weight: 700;
  cursor: pointer;
  background: rgba(251, 243, 227, .4);
  transition: border-color .12s ease, color .12s ease;
  min-height: 26px;
}
.slot-thumb-placeholder:hover {
  border-color: rgba(240, 42, 160, .4);
  color: rgba(240, 42, 160, .6);
}

.slot-footer {
  margin-top: 8px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.slot-vintage-label {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  font-weight: 600;
  color: #6B5B66;
  cursor: pointer;
}
.slot-vintage-label input { cursor: pointer; }

@media (max-width: 440px) {
  .slot-main-drop, .slot-main-photo { flex-basis: 34%; }
  .slot-badge { font-size: 10px; padding: 2px 7px; }
  .slot-meta { font-size: 10px; }
}
"""

UPLOAD_SLOT_DECK_JS = """
export default function(component) {
  const { data, parentElement, setTriggerValue } = component;
  const deckEl = parentElement.querySelector("[data-deck]");

  // Client-side-only "photo just selected, not yet confirmed by Python"
  // state. Declared at this outer scope (not inside render()) so it
  // survives across the repeated render() calls triggered by every new
  // `data` prop — Python doesn't know about these photos until their
  // add_files round trip lands.
  const pendingBySlot = new Map();   // slotIndex -> [{localId, objectUrl, file, status}]
  const lastRealCount = new Map();   // slotIndex -> photo count last seen from Python

  function emit(action) {
    setTriggerValue("action", action);
  }

  function isExternalFileDrag(event) {
    return !!(
      event.dataTransfer &&
      Array.from(event.dataTransfer.types || []).includes("Files")
    );
  }

  function makeLocalId() {
    return "local_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 8);
  }

  async function prepareFile(file) {
    const bitmap = await createImageBitmap(file);
    const MAX_SIDE = 1800;
    const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
    const width = Math.max(1, Math.round(bitmap.width * scale));
    const height = Math.max(1, Math.round(bitmap.height * scale));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d", { alpha: false });
    ctx.drawImage(bitmap, 0, 0, width, height);
    bitmap.close();
    const dataUrl = canvas.toDataURL("image/jpeg", 0.82);
    return {
      name: file.name.replace(/\\.[^.]+$/, "") + ".jpg",
      type: "image/jpeg",
      data: dataUrl
    };
  }

  // Drops "submitted" pending entries for a slot once Python's real
  // photo count for that slot has grown — the round trip landed, so the
  // real thumbnail takes over from the local preview. Doesn't touch
  // "uploading" (still in flight) or "failed" (needs retry/dismiss)
  // entries, and a count going down (a removal elsewhere) never clears
  // anything here.
  function reconcilePending(slots) {
    slots.forEach((slot) => {
      const slotIndex = Number(slot.slot_index);
      const realCount = Array.isArray(slot.photos) ? slot.photos.length : 0;
      const prevCount = lastRealCount.has(slotIndex) ? lastRealCount.get(slotIndex) : realCount;

      if (realCount > prevCount) {
        const pending = pendingBySlot.get(slotIndex) || [];
        const stillPending = [];
        pending.forEach((entry) => {
          if (entry.status === "submitted") {
            URL.revokeObjectURL(entry.objectUrl);
          } else {
            stillPending.push(entry);
          }
        });
        pendingBySlot.set(slotIndex, stillPending);
      }
      lastRealCount.set(slotIndex, realCount);
    });
  }

  // Encodes + emits one pending entry, independently of any siblings
  // selected alongside it — a large/slow photo never holds up the rest.
  async function processEntry(slotIndex, entry) {
    try {
      const prepared = await prepareFile(entry.file);
      entry.status = "submitted";
      emit({ type: "add_files", slot_index: slotIndex, files: [prepared] });
    } catch (error) {
      console.error("Photo processing failed", error);
      entry.status = "failed";
    }
    render();
  }

  function uploadFiles(slotIndex, files) {
    const valid = Array.from(files).filter(
      file => /\\.(jpe?g|png|webp)$/i.test(file.name)
    );
    if (!valid.length) return;

    const entries = valid.map((file) => ({
      localId: makeLocalId(),
      objectUrl: URL.createObjectURL(file),
      file,
      status: "uploading",
    }));

    const existing = pendingBySlot.get(slotIndex) || [];
    pendingBySlot.set(slotIndex, existing.concat(entries));

    // Instant local preview — createObjectURL is near-zero-cost, unlike
    // the createImageBitmap -> canvas resize -> toDataURL -> Python
    // round trip in processEntry() below, so this paints immediately
    // instead of waiting on that whole pipeline.
    render();

    entries.forEach((entry) => { processEntry(slotIndex, entry); });
  }

  function retryPendingEntry(slotIndex, entry) {
    entry.status = "uploading";
    render();
    processEntry(slotIndex, entry);
  }

  function dismissPendingEntry(slotIndex, entry) {
    const pending = pendingBySlot.get(slotIndex) || [];
    pendingBySlot.set(slotIndex, pending.filter((item) => item !== entry));
    URL.revokeObjectURL(entry.objectUrl);
    render();
  }

  function buildStatusOverlay(slotIndex, entry) {
    const overlay = document.createElement("div");
    if (entry.status === "failed") {
      overlay.className = "slot-pending-overlay is-failed";

      const label = document.createElement("div");
      label.className = "slot-pending-label";
      label.textContent = "Upload failed";
      overlay.appendChild(label);

      const retryBtn = document.createElement("button");
      retryBtn.type = "button";
      retryBtn.className = "slot-retry-btn";
      retryBtn.textContent = "Retry";
      retryBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        retryPendingEntry(slotIndex, entry);
      });
      overlay.appendChild(retryBtn);

      const dismissBtn = document.createElement("button");
      dismissBtn.type = "button";
      dismissBtn.className = "slot-dismiss-btn";
      dismissBtn.title = "Remove";
      dismissBtn.textContent = "×";
      dismissBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        dismissPendingEntry(slotIndex, entry);
      });
      overlay.appendChild(dismissBtn);
    } else {
      overlay.className = "slot-pending-overlay";
      overlay.innerHTML = '<div class="spinner"></div>';
    }
    return overlay;
  }

  function attachCardDropZone(card, slotIndex) {
    let depth = 0;
    card.addEventListener("dragenter", (event) => {
      if (!isExternalFileDrag(event)) return;
      event.preventDefault();
      depth += 1;
      card.classList.add("file-over");
    });
    card.addEventListener("dragover", (event) => {
      if (!isExternalFileDrag(event)) return;
      event.preventDefault();
      event.stopPropagation();
      event.dataTransfer.dropEffect = "copy";
      card.classList.add("file-over");
    });
    card.addEventListener("dragleave", (event) => {
      if (!isExternalFileDrag(event)) return;
      depth -= 1;
      if (depth <= 0) {
        depth = 0;
        card.classList.remove("file-over");
      }
    });
    card.addEventListener("drop", (event) => {
      if (!isExternalFileDrag(event)) return;
      event.preventDefault();
      event.stopPropagation();
      depth = 0;
      card.classList.remove("file-over");
      uploadFiles(slotIndex, event.dataTransfer.files || []);
    });
  }

  // Pointer + FLIP drag-reorder among a card's own thumbnails — the
  // exact technique already proven in ITEM_CARD_JS's photo-card
  // reorder, scoped here to one card's own .slot-thumb-grid only (no
  // cross-card dragging; that stays the item editor's job later).
  function attachThumbDrag(thumbCard, grid, slotIndex) {
    let pointerDrag = null;

    function gridThumbs() {
      return Array.from(grid.querySelectorAll(".slot-thumb"))
        .filter(el => el !== pointerDrag?.card && !el.classList.contains("reorder-placeholder"));
    }

    function flip(mutate) {
      const cards = gridThumbs();
      const first = new Map(cards.map(el => [el, el.getBoundingClientRect()]));
      mutate();
      cards.forEach(el => {
        const a = first.get(el);
        const b = el.getBoundingClientRect();
        const dx = a.left - b.left;
        const dy = a.top - b.top;
        if (Math.abs(dx) < 1 && Math.abs(dy) < 1) return;
        el.style.transition = "none";
        el.style.transform = `translate(${dx}px, ${dy}px)`;
        requestAnimationFrame(() => {
          el.style.transition = "transform 180ms cubic-bezier(.2,.8,.2,1)";
          el.style.transform = "translate(0,0)";
        });
      });
    }

    function movePlaceholder(event) {
      const drag = pointerDrag;
      if (!drag?.placeholder) return;
      const cards = gridThumbs();
      if (!cards.length) return;

      let closest = null;
      let best = Infinity;
      for (const el of cards) {
        const r = el.getBoundingClientRect();
        const d = Math.hypot(
          event.clientX - (r.left + r.width / 2),
          event.clientY - (r.top + r.height / 2)
        );
        if (d < best) {
          best = d;
          closest = { el, r };
        }
      }
      if (!closest) return;

      const r = closest.r;
      const before = event.clientX < r.left + r.width / 2;
      const current = Array.from(grid.children).indexOf(drag.placeholder);
      const target = Array.from(grid.children).indexOf(closest.el);
      const desired = before ? target : target + 1;

      if (desired === current || desired === current + 1) return;

      flip(() => {
        if (before) grid.insertBefore(drag.placeholder, closest.el);
        else grid.insertBefore(drag.placeholder, closest.el.nextSibling);
      });
    }

    function startRealDrag(event) {
      const drag = pointerDrag;
      if (!drag || drag.active) return;
      drag.active = true;

      const rect = drag.card.getBoundingClientRect();
      const ph = document.createElement("div");
      ph.className = "slot-thumb reorder-placeholder";
      ph.style.width = rect.width + "px";
      ph.style.height = rect.height + "px";
      drag.placeholder = ph;
      grid.insertBefore(ph, drag.card);

      drag.card.classList.add("pointer-dragging");
      drag.card.style.position = "fixed";
      drag.card.style.left = (event.clientX - drag.offsetX) + "px";
      drag.card.style.top = (event.clientY - drag.offsetY) + "px";
      drag.card.style.width = rect.width + "px";
      drag.card.style.height = rect.height + "px";
      drag.card.style.margin = "0";
      drag.card.style.transition = "none";
      drag.card.style.transform = "scale(1.08) rotate(1.2deg)";
      drag.card.style.pointerEvents = "none";
      drag.card.style.zIndex = "99999";
    }

    function pointerMove(event) {
      const drag = pointerDrag;
      if (!drag || drag.card !== thumbCard) return;

      const distance = Math.hypot(
        event.clientX - drag.startX,
        event.clientY - drag.startY
      );
      if (!drag.active) {
        if (distance < 6) return;
        startRealDrag(event);
      }

      event.preventDefault();
      event.stopPropagation();
      drag.card.style.left = (event.clientX - drag.offsetX) + "px";
      drag.card.style.top = (event.clientY - drag.offsetY) + "px";
      movePlaceholder(event);
    }

    function pointerUp(event) {
      const drag = pointerDrag;
      if (!drag || drag.card !== thumbCard) return;
      pointerDrag = null;
      if (!drag.active) return;

      event.preventDefault();
      event.stopPropagation();

      if (drag.placeholder?.parentNode) {
        drag.placeholder.parentNode.insertBefore(thumbCard, drag.placeholder);
        drag.placeholder.remove();
      }

      thumbCard.classList.remove("pointer-dragging");
      thumbCard.style.position = "";
      thumbCard.style.left = "";
      thumbCard.style.top = "";
      thumbCard.style.width = "";
      thumbCard.style.height = "";
      thumbCard.style.margin = "";
      thumbCard.style.transition = "";
      thumbCard.style.transform = "";
      thumbCard.style.pointerEvents = "";
      thumbCard.style.zIndex = "";

      const ids = Array.from(grid.querySelectorAll(".slot-thumb"))
        .map(el => String(el.dataset.id));
      emit({ type: "reorder", slot_index: slotIndex, ids });

      thumbCard.dataset.justDragged = "true";
      setTimeout(() => delete thumbCard.dataset.justDragged, 300);
    }

    thumbCard.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      if (event.target.closest(".slot-thumb-remove")) return;

      event.preventDefault();
      event.stopPropagation();

      const rect = thumbCard.getBoundingClientRect();
      pointerDrag = {
        card: thumbCard,
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        offsetX: event.clientX - rect.left,
        offsetY: event.clientY - rect.top,
        active: false,
        placeholder: null
      };

      try { thumbCard.setPointerCapture(event.pointerId); } catch {}
    });
    thumbCard.addEventListener("pointermove", pointerMove);
    thumbCard.addEventListener("pointerup", pointerUp);
    thumbCard.addEventListener("pointercancel", pointerUp);
  }

  function render() {
    let slots = [];
    try {
      slots = JSON.parse(data?.slots_json || "[]");
      if (!Array.isArray(slots)) slots = [];
    } catch {
      slots = [];
    }

    deckEl.innerHTML = "";

    reconcilePending(slots);

    slots.forEach((slot) => {
      const slotIndex = Number(slot.slot_index);
      const itemNumber = Number(slot.item_number);
      const realPhotos = Array.isArray(slot.photos) ? slot.photos : [];
      const pendingEntries = pendingBySlot.get(slotIndex) || [];
      // Local-only previews appended after Python-confirmed photos, so
      // they show up immediately without waiting on their own round
      // trip — see uploadFiles()/processEntry() above.
      const photos = realPhotos.concat(
        pendingEntries.map((entry) => ({
          id: entry.localId,
          name: entry.file.name,
          src: entry.objectUrl,
          pending: entry,
        }))
      );

      const card = document.createElement("div");
      card.className = "slot-card";
      card.dataset.slotIndex = String(slotIndex);

      const header = document.createElement("div");
      header.className = "slot-header";
      const badge = document.createElement("span");
      badge.className = "slot-badge";
      badge.textContent = "ITEM " + itemNumber;
      const meta = document.createElement("span");
      meta.className = "slot-meta";
      meta.textContent = photos.length + (photos.length === 1 ? " photo" : " photos");
      const clearBtn = document.createElement("button");
      clearBtn.type = "button";
      clearBtn.className = "slot-clear-btn";
      clearBtn.title = "Clear this item";
      clearBtn.textContent = "⋯";
      clearBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (!photos.length) return;
        emit({ type: "clear_item", slot_index: slotIndex });
      });
      header.appendChild(badge);
      header.appendChild(meta);
      header.appendChild(clearBtn);
      card.appendChild(header);

      const fileInput = document.createElement("input");
      fileInput.type = "file";
      fileInput.multiple = true;
      fileInput.accept = "image/jpeg,image/png,image/webp";
      fileInput.style.display = "none";
      fileInput.addEventListener("change", () => {
        const files = Array.from(fileInput.files || []);
        fileInput.value = "";
        if (!files.length) return;
        uploadFiles(slotIndex, files);
      });
      card.appendChild(fileInput);

      const body = document.createElement("div");
      body.className = "slot-body";

      if (!photos.length) {
        const mainDrop = document.createElement("div");
        mainDrop.className = "slot-main-drop";
        mainDrop.innerHTML =
          '<div class="slot-main-drop-icon">⬆</div>' +
          '<div class="slot-main-drop-title">Add Main Photo</div>' +
          '<div class="slot-main-drop-sub">Drag &amp; drop or click to upload</div>';
        mainDrop.addEventListener("click", () => fileInput.click());
        body.appendChild(mainDrop);

        const grid = document.createElement("div");
        grid.className = "slot-thumb-grid";
        for (let i = 0; i < 6; i++) {
          const ph = document.createElement("div");
          ph.className = "slot-thumb-placeholder";
          ph.textContent = "+";
          ph.addEventListener("click", () => fileInput.click());
          grid.appendChild(ph);
        }
        body.appendChild(grid);
      } else {
        const mainPhotoData = photos[0];
        const mainPhoto = document.createElement("div");
        mainPhoto.className = "slot-main-photo";
        if (mainPhotoData.pending) {
          mainPhoto.classList.add(mainPhotoData.pending.status === "failed" ? "is-failed" : "is-pending");
        }
        const mainImg = document.createElement("img");
        mainImg.src = mainPhotoData.src || "";
        mainImg.alt = mainPhotoData.name || "Main photo";
        mainPhoto.appendChild(mainImg);
        if (mainPhotoData.pending) {
          mainPhoto.appendChild(buildStatusOverlay(slotIndex, mainPhotoData.pending));
        } else {
          const mainBadge = document.createElement("div");
          mainBadge.className = "slot-main-badge";
          mainBadge.textContent = "MAIN PHOTO";
          mainPhoto.appendChild(mainBadge);
        }
        body.appendChild(mainPhoto);

        const grid = document.createElement("div");
        grid.className = "slot-thumb-grid";

        const thumbs = photos.slice(1, 7);
        thumbs.forEach((photo) => {
          const thumbCard = document.createElement("div");
          thumbCard.className = "slot-thumb";
          thumbCard.dataset.id = String(photo.id);

          const img = document.createElement("img");
          img.src = photo.src || "";
          img.alt = photo.name || "";
          img.draggable = false;
          thumbCard.appendChild(img);

          if (photo.pending) {
            // Not yet confirmed by Python — no id it can act on, so no
            // remove/promote/reorder until the round trip lands.
            thumbCard.classList.add(photo.pending.status === "failed" ? "is-failed" : "is-pending");
            thumbCard.appendChild(buildStatusOverlay(slotIndex, photo.pending));
          } else {
            const removeBtn = document.createElement("button");
            removeBtn.type = "button";
            removeBtn.className = "slot-thumb-remove";
            removeBtn.title = "Remove photo";
            removeBtn.textContent = "×";
            removeBtn.addEventListener("click", (event) => {
              event.preventDefault();
              event.stopPropagation();
              emit({ type: "remove", slot_index: slotIndex, id: String(photo.id) });
            });
            thumbCard.appendChild(removeBtn);

            thumbCard.addEventListener("click", (event) => {
              if (event.target.closest(".slot-thumb-remove")) return;
              if (thumbCard.dataset.justDragged === "true") {
                event.preventDefault();
                event.stopPropagation();
                return;
              }
              emit({ type: "promote_main", slot_index: slotIndex, id: String(photo.id) });
            });

            attachThumbDrag(thumbCard, grid, slotIndex);
          }

          grid.appendChild(thumbCard);
        });

        const remaining = 6 - thumbs.length;
        for (let i = 0; i < remaining; i++) {
          const ph = document.createElement("div");
          ph.className = "slot-thumb-placeholder";
          ph.textContent = "+";
          ph.addEventListener("click", () => fileInput.click());
          grid.appendChild(ph);
        }

        body.appendChild(grid);
      }

      card.appendChild(body);

      const footer = document.createElement("div");
      footer.className = "slot-footer";
      const label = document.createElement("label");
      label.className = "slot-vintage-label";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = !!slot.vintage;
      checkbox.addEventListener("change", () => {
        emit({ type: "vintage_toggle", slot_index: slotIndex, value: checkbox.checked });
      });
      label.appendChild(checkbox);
      label.appendChild(document.createTextNode(" Vintage"));
      footer.appendChild(label);
      card.appendChild(footer);

      attachCardDropZone(card, slotIndex);

      deckEl.appendChild(card);
    });
  }

  render();

  return () => {
    // No timers or global listeners to clean up.
  };
}
"""

if STREAMLIT_V2_AVAILABLE:
    upload_slot_deck = _streamlit_v2_component(
        "upload_slot_deck_v1",
        html=UPLOAD_SLOT_DECK_HTML,
        css=UPLOAD_SLOT_DECK_CSS,
        js=UPLOAD_SLOT_DECK_JS,
        isolate_styles=True,
    )
else:
    upload_slot_deck = None


def build_slots_payload(slots, batch_start, thumbnail_fn):
    """Build the JSON-ready payload the component's data.slots_json
    expects. thumbnail_fn(path) -> data-url string, so each caller can
    plug in its own (already-cached) thumbnail generator."""
    payload = []
    for slot_index, slot in enumerate(slots):
        photos = [
            {
                "id": str(Path(path).resolve()),
                "name": Path(path).name,
                "src": thumbnail_fn(path),
            }
            for path in slot.get("paths", [])
        ]
        payload.append({
            "slot_index": slot_index,
            "item_number": batch_start + slot_index,
            "vintage": bool(slot.get("vintage", False)),
            "photos": photos,
        })
    return payload


def save_slot_files(file_payloads, upload_dir, label):
    """Save files emitted by the component's add_files action.

    Each payload is {name, type, data} where data is a base64 data URL.
    upload_dir is created if needed; the per-directory counter lives in
    session_state keyed by the directory itself so two different
    callers (app.py's uploads/, description_generator.py's
    uploads/description_generator/) never share or collide on counts.
    """
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    counter_key = f"_upload_slot_counter::{upload_dir.resolve()}"
    counter = st.session_state.get(counter_key, 0)

    saved = []
    for payload in file_payloads or []:
        if not isinstance(payload, dict):
            continue

        name = Path(str(payload.get("name", "photo.jpg"))).name
        data_url = str(payload.get("data", ""))
        if "," not in data_url:
            continue

        encoded = data_url.split(",", 1)[1]
        try:
            raw = base64.b64decode(encoded, validate=True)
        except Exception:
            continue
        if not raw:
            continue

        counter += 1
        output = upload_dir / f"{label}_{counter}_{name}"
        with open(output, "wb") as handle:
            handle.write(raw)
        saved.append(output)

    st.session_state[counter_key] = counter
    return saved


def handle_upload_slot_action(slots, action, upload_dir):
    """
    Apply an action emitted by the upload_slot_deck component to a
    caller-owned list of slot dicts (each with at least "paths" and
    "vintage" keys). Returns True if the slots were mutated.
    """
    if not isinstance(action, dict):
        return False

    action_type = action.get("action")

    try:
        slot_index = int(action.get("slot_index", -1))
    except (TypeError, ValueError):
        return False

    if slot_index < 0 or slot_index >= len(slots):
        return False

    slot = slots[slot_index]

    if action_type == "add_files":
        saved_paths = save_slot_files(
            action.get("files", []),
            upload_dir,
            f"slot_{slot_index + 1}",
        )

        if not saved_paths:
            return False

        existing = {
            str(Path(path).resolve())
            for path in slot.get("paths", [])
        }

        for path in saved_paths:
            key = str(Path(path).resolve())
            if key in existing:
                continue
            slot.setdefault("paths", []).append(path)
            existing.add(key)

        return True

    if action_type == "promote_main":
        photo_id = str(action.get("id", ""))
        paths = slot.get("paths", [])

        for index, path in enumerate(paths):
            if str(Path(path).resolve()) != photo_id:
                continue
            if index == 0:
                return False
            paths.pop(index)
            paths.insert(0, path)
            slot["paths"] = paths
            return True

        return False

    if action_type == "reorder":
        ids = action.get("ids", [])
        if not isinstance(ids, list):
            return False

        current = {
            str(Path(path).resolve()): path
            for path in slot.get("paths", [])
        }

        # A reorder from the component only ever carries the thumbnail
        # ids (everything except the main photo at index 0) — validate
        # it's an exact permutation of exactly that subset, then splice
        # it back in after the untouched main photo.
        main_path = slot.get("paths", [None])[0] if slot.get("paths") else None
        thumb_ids = {
            key for key in current
            if main_path is None or key != str(Path(main_path).resolve())
        }

        if set(ids) != thumb_ids or len(ids) != len(thumb_ids):
            return False

        reordered = [current[path_id] for path_id in ids]
        slot["paths"] = ([main_path] if main_path is not None else []) + reordered
        return True

    if action_type == "remove":
        photo_id = str(action.get("id", ""))
        old_paths = slot.get("paths", [])

        new_paths = [
            path for path in old_paths
            if str(Path(path).resolve()) != photo_id
        ]

        if len(new_paths) == len(old_paths):
            return False

        slot["paths"] = new_paths
        return True

    if action_type == "vintage_toggle":
        slot["vintage"] = bool(action.get("value", False))
        return True

    if action_type == "clear_item":
        slots[slot_index] = {
            "slot": slot_index,
            "paths": [],
            "signature": [],
            "vintage": False,
        }
        return True

    return False
