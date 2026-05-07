/// <reference path="../.astro/types.d.ts" />
/// <reference types="astro/client" />

type Runtime = import('@astrojs/cloudflare').Runtime<{
  DB: D1Database;
  ASSETS: Fetcher;
  // BACKUPS: R2Bucket;  // restored once R2 is enabled
}>;

declare namespace App {
  interface Locals extends Runtime {}
}
