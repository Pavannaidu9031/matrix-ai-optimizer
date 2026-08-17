const CACHE_NAME = 'matrixai-v1';
const ASSETS = ['/', '/static/manifest.json', 'https://cdn.tailwindcss.com', 'https://cdn.plot.ly/plotly-2.27.0.min.js'];

self.addEventListener('install', (e) => {
    e.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS)));
});

self.addEventListener('fetch', (e) => {
    e.respondWith(caches.match(e.request).then((res) => res || fetch(e.request)));
});
