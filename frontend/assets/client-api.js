"use strict";

(() => {
  let worker;
  let nextId = 1;
  const waiting = new Map();
  function getWorker() {
    if (worker) return worker;
    if (!("Worker" in window)) throw new Error("This browser does not support the local computation worker.");
    worker = new Worker("/assets/client-worker.js?v=20261122-3");
    worker.onmessage = ({ data }) => {
      const promise = waiting.get(data.id);
      if (!promise) return;
      waiting.delete(data.id);
      if (data.error) promise.reject(new Error(data.error));
      else promise.resolve(data.result);
    };
    worker.onerror = event => {
      const message = event.message || "The browser computation worker stopped unexpectedly.";
      for (const promise of waiting.values()) promise.reject(new Error(message));
      waiting.clear();
      worker?.terminate();
      worker = null;
    };
    return worker;
  }
  async function request(path, options = {}) {
    return new Promise((resolve, reject) => {
      let active;
      try { active = getWorker(); }
      catch (error) { reject(error); return; }
      const id = nextId++;
      waiting.set(id, { resolve, reject });
      active.postMessage({ id, path, options });
    });
  }
  window.CreditViewBrowser = { request };
})();
