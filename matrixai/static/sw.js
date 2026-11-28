const CACHE_NAME = 'matrixai-v1';
const ASSETS = ['/', '/static/manifest.json', 'https://cdn.tailwindcss.com', 'https://cdn.plot.ly/plotly-2.27.0.min.js'];

self.addEventListener('install', (e) => {
    e.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS)));
});

self.addEventListener('fetch', (e) => {
    // Personalized pages must use the current authenticated session.
    if (e.request.mode === 'navigate') {
        e.respondWith(fetch(e.request, { cache: 'no-store' }));
        return;
    }
    e.respondWith(caches.match(e.request).then((res) => res || fetch(e.request)));
});

// Refresh the cached landing page when the MatrixAI identity update activates.
self.addEventListener('activate', (e) => {
    e.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => cache.add(new Request('/', { cache: 'reload' })))
            .catch(() => { /* Keep the existing offline page if the network is unavailable. */ })
    );
});
