import path from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { pipelineApi } from "./vite-plugins/pipeline.ts";

// This config file lives in frontend/, so the repo root (where `make`
// commands run) is one directory up.
const repoRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), pipelineApi(repoRoot)],
})
