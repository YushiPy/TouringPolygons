import { chromium, webkit } from '@playwright/test';
const browser = await (process.env.TEST_BROWSER === "webkit" ? webkit : chromium).launch({headless: true});
try {
 const page = await browser.newPage({deviceScaleFactor:2,viewport:{width:1440,height:1000}});
 page.on('response', async r=>{if(r.url().includes('/api/editor/solve'))console.log('SOLVE',r.status(),await r.text())});
 page.on('pageerror', e=>console.log('ERROR',e.message));
 await page.addInitScript(() => { window.sizes=[]; for(const prop of ['width','height']) { const descriptor=Object.getOwnPropertyDescriptor(HTMLCanvasElement.prototype,prop); Object.defineProperty(HTMLCanvasElement.prototype,prop,{...descriptor,set(value){window.sizes.push([this.id,prop,value]);if(value>8192)throw new Error('Runaway canvas '+this.id+' '+prop+' '+value);descriptor.set.call(this,value)}}); } });
 await page.goto('http://127.0.0.1:8017/');
 await page.waitForFunction(()=>window.__benchmarkDashboardReady);
 await page.getByRole('button',{name:'Cases',exact:true}).click();
 await page.locator('#manual-visit-order').selectOption('free');
 await page.waitForTimeout(1500);
 console.log(await page.locator('body').innerText());
}finally{await browser.close()}
