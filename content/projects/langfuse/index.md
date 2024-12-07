+++
title = "A open-source contributor at Langfuse"
date = "2024-08-01"
github = "https://github.com/langfuse/langfuse/pull/2866"
+++

Add authenticate option to support KeyCloak

<!--more-->
[GitHub Repository](https://github.com/RTae/yolov11-cuda)

# Feature

- Add Environment variable for KeyCloak authentication in web/src/env.mjs and .env.prod.example
- Add Keycloak provider schema type in ee/src/sso/types.ts
-  Add a sign-in button for Keycloak when providing a Keycloak configuration in web/src/pages/auth/sign-in.tsx
- Add a function to push auth provider into list of auth provider when KeyCloak's configuration is set in web/src/server/auth.ts
-  Add a logic to delete refresh_expires_in and not-before-policy before processing a linking account, since Keycloak returns payload data that is incompatible with current nextjs-auth schema.


# Report