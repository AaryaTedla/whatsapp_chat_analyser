// Service Worker for Lumira PWA
const CACHE_NAME = 'lumira-v2.0.0';
const STATIC_ASSETS = [
  '/',
  '/static/manifest.json',
  'https://cdn.jsdelivr.net/npm/chart.js@latest',
];

// Install event
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(STATIC_ASSETS).catch(err => {
        console.log('Failed to cache static assets:', err);
      });
    })
  );
  self.skipWaiting();
});

// Activate event
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
  self.clients.claim();
});

// Fetch event -  Network first, fallback to cache
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip cross-origin and API requests with custom logic
  if (url.origin !== self.location.origin) {
    return;
  }

  // API calls: network first
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      fetch(request)
        .then(response => {
          if (response.ok) {
            const cache_copy = response.clone();
            caches.open(CACHE_NAME).then(cache => {
              cache.put(request, cache_copy);
            });
          }
          return response;
        })
        .catch(() => {
          return caches.match(request) || caches.match('/');
        })
    );
    return;
  }

  // Static assets: cache first
  event.respondWith(
    caches.match(request).then(cachedResponse => {
      if (cachedResponse) {
        return cachedResponse;
      }
      return fetch(request).then(response => {
        if (!response || response.status !== 200) {
          return response;
        }
        const responseClone = response.clone();
        caches.open(CACHE_NAME).then(cache => {
          cache.put(request, responseClone);
        });
        return response;
      }).catch(() => {
        return caches.match('/');
      });
    })
  );
});

// Background sync for offline reports
self.addEventListener('sync', event => {
  if (event.tag === 'sync-reports') {
    event.waitUntil(syncReports());
  }
});

async function syncReports() {
  try {
    const db = await openDB('lumira-cache');
    const pendingReports = await db.getAll('pendingReports');
    
    for (let report of pendingReports) {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: JSON.stringify(report),
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        await db.delete('pendingReports', report.id);
      }
    }
  } catch (error) {
    console.log('Sync failed:', error);
    throw error;
  }
}

// Notifications
self.addEventListener('push', event => {
  const data = event.data ? event.data.json() : {};
  const options = {
    body: data.body || 'Your analysis is ready!',
    icon: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 192 192"><rect fill="%237c3aed" width="192" height="192"/><text x="96" y="145" text-anchor="middle" font-size="120" font-weight="bold" fill="white">L</text></svg>',
    badge: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 192 192"><rect fill="%237c3aed" width="192" height="192"/></svg>',
    tag: 'lumira-notification',
    requireInteraction: false,
  };
  
  event.waitUntil(self.registration.showNotification('Lumira', options));
});

self.addEventListener('notificationclick', event => {
  event.notification.close();
  event.waitUntil(
    clients.matchAll({ type: 'window' }).then(clientList => {
      if (clientList.length > 0) {
        return clientList[0].focus();
      }
      return clients.openWindow('/');
    })
  );
});
