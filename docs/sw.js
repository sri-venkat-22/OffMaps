// Service worker: the page and the engine work offline once visited (and install as an app).
//   page (navigations)   network first, cached copy when offline -- a new deploy shows at once
//   everything else      cache first, refreshed in the background (stale-while-revalidate)
// SHELL is fetched at install so the replay and the in-browser engine run with no network;
// the Hyderabad scenario, fonts and a sample drive are cached the first time they are used.
// Road matching in "Run it in your browser" needs the network (OpenStreetMap), by design.
const CACHE = "offmaps-v8";
const SHELL = [
  "./", "index.html", "style.css?v=7", "app.js?v=7", "live.js?v=7", "manifest.webmanifest",
  "icon-192.png", "icon-512.png", "icon-maskable-512.png", "apple-touch-icon.png",
  "engine/core.js", "engine/aids.js", "engine/speednet.js", "engine/head.js", "engine/hmm.js",
  "engine/engine.js", "engine/csv.js", "engine/worker.js",
  "engine/model/speednet.json", "engine/model/speednet.bin", "engine/model/profile.json", "engine/model/fusion_head.json",
  "data/ablation.json", "data/outages.json", "data/roads.json",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const req = e.request, url = new URL(req.url);
  if (req.method !== "GET") return;                               // e.g. the OpenStreetMap road query
  const same = url.origin === location.origin;
  const fonts = url.hostname === "fonts.googleapis.com" || url.hostname === "fonts.gstatic.com";
  if (!same && !fonts) return;
  if (req.mode === "navigate") {
    // revalidate with the server: the browser's HTTP cache may hold a page from before a deploy
    e.respondWith(fetch(req.url, { cache: "no-cache", credentials: "same-origin" })
      .then((r) => { const c = r.clone(); if (r.ok) caches.open(CACHE).then((x) => x.put(req, c)); return r; })
      .catch(() => caches.match(req).then((r) => r || caches.match("index.html"))));
    return;
  }
  e.respondWith(caches.open(CACHE).then((c) => c.match(req).then((hit) => {
    const net = fetch(req).then((r) => { if (r.ok || r.type === "opaque") c.put(req, r.clone()); return r; }).catch(() => hit);
    return hit || net;
  })));
});
