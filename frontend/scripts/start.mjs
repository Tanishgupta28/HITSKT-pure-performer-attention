import { cpSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const server = resolve(".next/standalone/server.js");
if (!existsSync(server))
  throw new Error("Run npm run build before starting the production server.");
cpSync(resolve(".next/static"), resolve(".next/standalone/.next/static"), {
  recursive: true,
});
process.env.HOSTNAME ||= "127.0.0.1";
await import(pathToFileURL(server).href);
