/* WORKBUDDY_LOCAL_MODELS_V2 */
var __wbLocalOnly = (function createWorkBuddyLocalPolicy() {
  function fail(message) {
    const error = new Error('[WorkBuddy local-only] ' + message);
    error.code = 'WORKBUDDY_MODEL_POLICY_DENIED';
    throw error;
  }
  function localURL(value, base) {
    let url;
    try { url = base ? new URL(value, base) : new URL(value); }
    catch { fail('Invalid local inference URL.'); }
    const host = url.hostname.toLowerCase();
    if (!['http:', 'https:'].includes(url.protocol) ||
        !(host === 'localhost' || host === '[::1]' || /^127(?:\.\d{1,3}){3}$/.test(host)) ||
        url.username || url.password) {
      fail('Inference must use a loopback endpoint; cloud fallback is disabled.');
    }
    return url;
  }
  function assertModel(value) {
    if (typeof value !== 'string' || !value.trim()) fail('Select a configured local model.');
  }
  function localConfig(config) {
    if (!config || config.disabled) fail('No enabled local model is configured.');
    assertModel(config.id);
    localURL(config.url);
    // Preserve provider protocol, model ID, aliases, capabilities and token budgets.
    return {...config, tags: [...new Set([...(config.tags || []), 'custom'])]};
  }
  function filterModels(models) {
    return (models || []).flatMap(config => {
      try { return [localConfig(config)]; } catch { return []; }
    });
  }
  function matches(config, value) {
    return typeof value === 'string' && !!value && (config.id === value || config.name === value || config.aliases?.includes(value));
  }
  function currentId(models, preferred) {
    const available = filterModels(models);
    return (available.find(config => matches(config, preferred)) || available[0])?.id;
  }
  async function select(manager, agentName, session) {
    const available = filterModels([...manager.modelMap.values()]);
    const agent = manager.agentConfigMap?.get(agentName);
    // Resolve afresh for each call, so runtime session and model-list changes take effect.
    const preferences = [session?.requestOptions?.model, session?.options?.model,
      process.env.CODEBUDDY_MODEL, await manager.settingsManager.get('model'),
      agent?.declaredModel, ...(agent?.models || [])];
    for (const value of preferences) {
      const match = available.find(config => matches(config, value));
      if (match) return match;
    }
    if (available.length) return available[0];
    fail('No enabled model with a local endpoint is configured; cloud fallback is disabled.');
  }
  function resolveBaseURL(options) {
    const base = options.modelConfigUrl || options.envBaseURL;
    localURL(base);
    return base;
  }
  function guardRequest(request) {
    localURL(request.url, request.baseURL);
    request.maxRedirects = 0;
    request.proxy = false;
    const transforms = request.transformRequest;
    request.transformRequest = [...(Array.isArray(transforms) ? transforms : transforms ? [transforms] : []), function(data) {
      localURL(this.url, this.baseURL);
      this.maxRedirects = 0;
      this.proxy = false;
      return data;
    }];
    return request;
  }
  return Object.freeze({assertModel, filterModels, localConfig, currentId, select, resolveBaseURL, guardRequest});
})();
