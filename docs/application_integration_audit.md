# Frontend/backend integration audit

## Initial findings

- Dashboard cards contained hardcoded Late Blight, Tomato, severity, confidence, crop-confidence, dates, health percentage, active-disease counts, treatment instructions, and chart percentages.
- History inserted sample records when the database returned an empty list.
- The recent-detection card displayed a stock diseased-leaf image and a fabricated confidence when no scan existed.
- Search, notification, edit, zoom, share, treatment actions, and multiple sidebar destinations had no implementation.
- `/predict` used the released ONNX model correctly, but scans and feedback were global instead of user-owned.
- No dashboard aggregation, users, sessions, OAuth state, user ownership, or CSRF enforcement existed.
- Unreviewed disease classes returned placeholder guidance instead of an explicit unavailable state.

## Implemented resolution

- The application exposes only working Dashboard, New scan, Scan history, Profile, feedback, and logout controls.
- All statistics, charts, recent scans, confidence summaries, and history records come from authenticated SQLite queries through `/dashboard` and `/history`.
- Empty accounts render empty states; failed requests render errors; pending requests render loading states. No record or prediction is generated in the browser.
- Predictions, batch predictions, history, dashboard data, and feedback are protected and scoped by `user_id`.
- Google OAuth profiles populate `users`; opaque sessions and OAuth state are stored separately. Google tokens are not stored.
- Upload validation accepts only decoded JPEG, PNG, or WebP files within the size/pixel limits. Low-detail images are rejected, while dark, bright, low-contrast, or blurred images return explicit quality warnings.
- ONNX preprocessing and class order continue to come from the immutable EfficientNetV2-S v1 metadata. The release bytes and checksum were not changed.
- Only reviewed disease guidance is stored. Missing guidance is returned as `information_status: unavailable` and rendered as an explicit empty guidance state.
