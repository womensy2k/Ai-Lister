// Service worker for Y2K Lister's push notifications. Lives on GitHub
// Pages (this app's manifest/start_url origin — see index.html and
// notifications.html for why), NOT on the Streamlit-hosted app itself.
// Registered with scope "./" so it can receive push events regardless
// of which page under this origin subscribed.

self.addEventListener("install", function (event) {
  self.skipWaiting();
});

self.addEventListener("activate", function (event) {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("push", function (event) {
  var payload = {};
  try {
    payload = event.data ? event.data.json() : {};
  } catch (error) {
    payload = {};
  }

  var title = payload.title || "Y2K Lister";
  var body = payload.body || "";
  var url = payload.url || "https://y2klister.streamlit.app/?embed=true&embed_options=light_theme";

  event.waitUntil(
    self.registration.showNotification(title, {
      body: body,
      icon: "icon-192.png",
      badge: "icon-192.png",
      data: { url: url },
    })
  );
});

self.addEventListener("notificationclick", function (event) {
  event.notification.close();
  var url =
    (event.notification.data && event.notification.data.url) ||
    "https://y2klister.streamlit.app/?embed=true&embed_options=light_theme";

  event.waitUntil(
    self.clients
      .matchAll({ type: "window", includeUncontrolled: true })
      .then(function (windowClients) {
        for (var i = 0; i < windowClients.length; i++) {
          var client = windowClients[i];
          if ("focus" in client) {
            if ("navigate" in client) {
              client.navigate(url);
            }
            return client.focus();
          }
        }
        if (self.clients.openWindow) {
          return self.clients.openWindow(url);
        }
      })
  );
});
