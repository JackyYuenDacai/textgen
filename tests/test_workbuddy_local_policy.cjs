const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const stage = process.argv[2];
async function verify(file) {
  const source = fs.readFileSync(file, 'utf8');
  const end = source.indexOf('\n})();') + '\n})();'.length;
  const sandbox = {URL, process: {env: {}}};
  vm.createContext(sandbox);
  vm.runInContext(source.slice(0,end), sandbox);
  const p = sandbox.__wbLocalOnly;
  const a = {id:'custom-local:Qwen',name:'Qwen',url:'http://127.0.0.1:8317/v1',maxInputTokens:262144};
  const b = {id:'custom:Bonsai2\\model.gguf',name:'Bonsai 2',url:'http://localhost:9000/api/v1/',aliases:['bonsai'],supportsToolCall:true};
  const c = {id:'another-model',url:'http://[::1]:8080/v1',useCustomProtocol:true};
  const cloud = {id:'cloud',url:'https://example.com/v1'};
  const disabled = {...a,id:'disabled',disabled:true};
  assert.equal(p.filterModels([a,b,c,cloud,disabled]).length,3);
  assert.equal(p.localConfig(b).id,b.id);
  assert.equal(p.localConfig(b).url,b.url);
  assert.equal(p.localConfig(a).maxInputTokens,262144);
  assert.equal(p.localConfig(c).useCustomProtocol,true);
  for (const url of ['https://example.com/v1','http://localhost.evil.test/v1','http://127.0.0.1@evil.test/v1','file:///tmp/a'])
    assert.throws(()=>p.resolveBaseURL({modelConfigUrl:url}));
  assert.equal(p.resolveBaseURL({modelConfigUrl:b.url,envBaseURL:a.url}),b.url);
  assert.equal(p.resolveBaseURL({envBaseURL:a.url}),a.url);
  assert.throws(()=>p.resolveBaseURL({productEndpoint:'https://cloud.example'}));
  const manager = {modelMap:new Map([a,b,c,cloud,disabled].map(x=>[x.id,x])),agentConfigMap:new Map(),settingsManager:{get:async()=>a.id}};
  assert.equal((await p.select(manager,'main',{requestOptions:{model:b.id},options:{model:a.id}})).id,b.id);
  assert.equal((await p.select(manager,'main',{options:{model:'bonsai'}})).id,b.id);
  assert.equal((await p.select(manager,'main')).id,a.id);
  manager.settingsManager.get=async()=>b.id;
  assert.equal((await p.select(manager,'main')).id,b.id);
  manager.modelMap.delete(b.id);
  assert.equal((await p.select(manager,'main',{requestOptions:{model:b.id}})).id,a.id);
  manager.modelMap.delete(a.id);
  assert.equal((await p.select(manager,'main')).id,c.id);
  manager.modelMap.delete(c.id);
  await assert.rejects(()=>p.select(manager,'main'),/No enabled model/);
  assert.equal(p.currentId([a,b],'bonsai'),b.id);
  assert.equal(p.currentId([a,b],'removed-model'),a.id);
  for (const [endpoint,method,data] of [['/v1/chat/completions','POST',{model:b.id}],['/v1/responses','POST',{model:'any-future-model'}],['/v1/models','GET',undefined],['/api/messages','POST','raw-payload']]) {
    const request={url:'http://127.0.0.1:9999'+endpoint,method,data};
    p.guardRequest(request);
    assert.equal(request.proxy,false);assert.equal(request.maxRedirects,0);
    assert.equal(request.transformRequest.at(-1).call(request,data),data);
    request.url='https://cloud.example/v1';
    assert.throws(()=>request.transformRequest.at(-1).call(request,data));
  }
  const relative={baseURL:'http://localhost:9001/v1/',url:'responses',method:'POST',data:'{}'};
  p.guardRequest(relative);assert.equal(relative.transformRequest.at(-1).call(relative,'{}'),'{}');
  assert.throws(()=>p.guardRequest({url:'https://example.com/v1/chat/completions'}));
  // Exercise the actual injected AgentManager entry point, not only its helper.
  const compute=source.slice(source.indexOf('async computeModel(eA){'),source.indexOf('async getRelatedModel(eA,el){'));
  sandbox.manager={...manager,modelMap:new Map([[a.id,a],[b.id,b]]),settingsManager:{get:async()=>b.id}};
  const chosen=await vm.runInContext('(new (class {'+compute+'})).computeModel.call(manager,"main")',sandbox);
  assert.equal(chosen.id,b.id);
  assert.ok(source.includes('let ec=await __wbLocalOnly.select(this.agentManager,eA.name,el);'));
  assert.ok(source.includes('currentModelId:__wbLocalOnly.currentId(eu,el)'));
  assert.ok(!source.includes('__wbQwenOnly'));
  console.log(path.basename(file)+': selection, live switching, limits, local endpoints, and cloud rejection passed');
}
(async()=>{for(const name of ['codebuddy.js','codebuddy-headless.js'])await verify(path.join(stage,name));})().catch(e=>{console.error(e);process.exitCode=1;});
