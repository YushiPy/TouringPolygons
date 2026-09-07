import { escapeHTML } from "./dom.js";
import { convexHull, filterRows, gapRatio, pathPrefix, projectedCase } from "./event-geometry.js";

const element = (id) => document.getElementById(id);
const number = (value, digits = 4) => value.toLocaleString("pt-BR", { maximumFractionDigits: digits });
const percent = (value) => value < 0.000001 ? "< 0,0001%" : `${number(value * 100)}%`;
const coordinates = (points) => points.map((point) => point.join(",")).join(" ");
const titles = { 2: "Quatro regiões, um caminho", 9: "Quarenta regiões, ordem livre", 55: "Um caminho, uma prova em aberto" };

function download(name, text, type) {
	const url = URL.createObjectURL(new Blob([text], { type }));
	const link = document.createElement("a");
	link.href = url;
	link.download = name;
	document.body.append(link);
	link.click();
	link.remove();
	setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function initialize() {
	const data = JSON.parse(element("event-data").textContent);
	if (data.schema_version !== 1 || !data.rows.length) throw new Error("Dados da demonstração indisponíveis.");
	let row = data.rows.find((item) => item.case === 2);
	let projected, fraction = 1, zoom = 1, frame = 0, playing = false;
	let pan = [0, 0], drag = null;
	const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
	const mapContent = element("map-content");
	const map = element("route-map");

	function camera() {
		pan = [Math.max(-420 * (zoom - 1), Math.min(420 * (zoom - 1), pan[0])), Math.max(-240 * (zoom - 1), Math.min(240 * (zoom - 1), pan[1]))];
		mapContent.setAttribute("transform", `translate(${420 + pan[0]} ${240 + pan[1]}) scale(${zoom}) translate(-420 -240)`);
		map.classList.toggle("is-zoomed", zoom > 1);
		map.style.touchAction = zoom > 1 ? "none" : "pan-y";
	}

	function stop() {
		cancelAnimationFrame(frame);
		playing = false;
		element("play-route").textContent = "▶ Percorrer caminho";
		element("play-route").setAttribute("aria-pressed", "false");
	}

	function drawRoute() {
		const prefix = pathPrefix(projected.path, fraction);
		element("animated-route").setAttribute("points", coordinates(prefix));
		const point = prefix.at(-1);
		element("traveler").setAttribute("cx", point[0]);
		element("traveler").setAttribute("cy", point[1]);
		element("traveler").style.display = fraction >= 1 ? "none" : "";
		element("route-progress").value = Math.round(fraction * 1000);
		element("progress-value").textContent = `${Math.round(fraction * 100)}%`;
	}

	function draw() {
		projected = projectedCase(row);
		const start = projected.project(row.geometry.start), target = projected.project(row.geometry.target);
		const labels = element("show-labels").checked;
		const hulls = element("show-hulls").checked;
		mapContent.innerHTML = `${hulls ? row.geometry.polygons.map((polygon) => `<polygon class="hull" points="${coordinates(convexHull(polygon).map(projected.project))}"/>`).join("") : ""}
			${projected.polygons.map((polygon, index) => `<polygon class="region" data-region="${index}" points="${coordinates(polygon)}"><title>Região ${index + 1}; ${row.order.indexOf(index) + 1}ª na ordem exportada</title></polygon>`).join("")}
			<polyline class="route-ghost" points="${coordinates(projected.path)}"/><polyline id="animated-route" class="route-line"/>
			${labels ? projected.polygons.map((polygon, index) => {
				const center = polygon.reduce((sum, point) => [sum[0] + point[0] / polygon.length, sum[1] + point[1] / polygon.length], [0, 0]);
				return `<text class="region-label" x="${center[0]}" y="${center[1]}" text-anchor="middle">${index + 1}</text>`;
			}).join("") : ""}
			<circle class="endpoint" cx="${start[0]}" cy="${start[1]}" r="7"/><text class="endpoint-label" x="${start[0] + 14}" y="${start[1] + 5}">S</text>
			<circle class="endpoint target" cx="${target[0]}" cy="${target[1]}" r="7"/><text class="endpoint-label" x="${target[0] + 14}" y="${target[1] + 5}">T</text><circle id="traveler" class="traveler" r="6"/>`;
		const rect = map.getBoundingClientRect();
		const textScale = 1 / Math.max(.1, Math.min(rect.width / 840, rect.height / 480)) / zoom;
		mapContent.querySelectorAll(".region-label").forEach((label) => { label.style.fontSize = `${13 * textScale}px`; });
		mapContent.querySelectorAll(".endpoint-label").forEach((label, index) => {
			const point = index ? target : start;
			label.style.fontSize = `${16 * textScale}px`;
			label.setAttribute("x", point[0] + 14 * textScale);
			label.setAttribute("y", point[1] + 5 * textScale);
		});
		mapContent.querySelectorAll(".endpoint, .traveler").forEach((point) => point.setAttribute("r", 6 * textScale));
		camera();
		element("map-caption").textContent = hulls ? "Os contornos tracejados são fechos convexos: regiões maiores que simplificam o problema para obter limites inferiores." : "Basta tocar a borda ou atravessar a região. Seus centros não são pontos de visita obrigatórios.";
		element("map-title").textContent = `Caso ${row.case}: caminho verificado por ${row.polygons} regiões; ${row.exact ? "gap numérico fechado" : "limite de tempo"}.`;
		element("zoom-in").disabled = zoom >= 3;
		element("zoom-out").disabled = zoom <= 1;
		drawRoute();
	}

	function selectCase(index, updateURL = true) {
		const selected = data.rows.find((item) => item.case === index);
		if (!selected) return false;
		stop();
		row = selected;
		fraction = 1;
		zoom = 1;
		pan = [0, 0];
		element("show-labels").checked = row.polygons <= 15;
		element("case-select").value = row.case;
		document.querySelectorAll(".example").forEach((button) => {
			const active = Number(button.dataset.case) === row.case;
			button.classList.toggle("active", active);
			button.setAttribute("aria-pressed", String(active));
		});
		element("drawing-title").textContent = titles[row.case] || `${row.polygons} regiões, extremos fixos`;
		element("case-id").textContent = `Caso ${String(row.case).padStart(2, "0")}`;
		element("outcome-badge").className = `status ${row.exact ? "certified" : "limited"}`;
		element("outcome-badge").textContent = row.exact ? "✓ Gap numérico fechado" : "◷ Encerrado por tempo";
		element("outcome-title").textContent = row.exact ? "Um caminho ótimo nas tolerâncias." : "Caminho viável. O ótimo segue em aberto.";
		element("outcome-explanation").textContent = row.exact ? "Os limites inferior e superior ficaram suficientemente próximos para encerrar a busca." : "O caminho visita todas as regiões. A diferença entre os limites ainda não permite certificar o ótimo.";
		element("case-length").textContent = number(row.upper_bound);
		element("case-time").textContent = row.seconds < .001 ? "< 0,001 s" : `${number(row.seconds, 3)} s`;
		element("case-regions").textContent = `${row.polygons} / ${row.polygons}`;
		element("case-gap").textContent = percent(gapRatio(row));
		element("case-lower").textContent = row.lower_bound.toPrecision(17);
		element("case-upper").textContent = row.upper_bound.toPrecision(17);
		element("case-order").textContent = row.order.map((index) => index + 1).join(" → ");
		element("case-calls").textContent = `${number(row.calls, 0)} chamadas ao oráculo convexo; ${number(row.fallback_calls, 0)} ao método auxiliar. Distância máxima às regiões: ${row.validation.max_polygon_distance.toExponential(2)}. Hash da instância: ${row.sha256}.`;
		if (updateURL && window.location.protocol !== "file:") {
			const url = new URL(window.location.href);
			url.searchParams.set("caso", row.case);
			window.history.replaceState(null, "", url);
		}
		draw();
		return true;
	}

	function renderTable() {
		const rows = filterRows(data.rows, element("result-filter").value, element("case-search").value);
		element("result-count").textContent = `${rows.length} de ${data.rows.length} resultados. Selecione um caso para ver o caminho.`;
		element("result-rows").innerHTML = rows.length ? rows.map((item) => `<tr><th scope="row">${String(item.case).padStart(2, "0")}</th><td>${item.polygons}</td><td>${item.seconds < .001 ? "< 0,001" : number(item.seconds, 3)}</td><td>${escapeHTML(percent(gapRatio(item)))}</td><td><span class="status ${item.exact ? "certified" : "limited"}">${item.exact ? "✓ Gap fechado" : "◷ Limite de tempo"}</span></td><td><button type="button" data-open-case="${item.case}" aria-label="Ver caminho do caso ${item.case}">Ver caminho ↗</button></td></tr>`).join("") : '<tr><td colspan="6">Nenhum caso corresponde à busca. Limpe o número ou escolha “Todos os resultados”.</td></tr>';
	}

	document.querySelectorAll("button:disabled, input:disabled, select:disabled").forEach((control) => { control.disabled = false; });
	document.querySelectorAll(".example").forEach((button) => button.addEventListener("click", () => selectCase(Number(button.dataset.case))));
	element("case-select").addEventListener("change", (event) => selectCase(Number(event.target.value)));
	element("show-labels").addEventListener("change", draw);
	element("show-hulls").addEventListener("change", draw);
	element("zoom-in").addEventListener("click", () => { zoom = Math.min(3, zoom + .5); draw(); });
	element("zoom-out").addEventListener("click", () => { zoom = Math.max(1, zoom - .5); draw(); });
	element("fit-view").addEventListener("click", () => { zoom = 1; pan = [0, 0]; draw(); });
	map.addEventListener("pointerdown", (event) => {
		if (zoom <= 1 || event.button !== 0) return;
		drag = [event.clientX, event.clientY, ...pan];
		map.setPointerCapture(event.pointerId);
		map.classList.add("is-dragging");
	});
	map.addEventListener("pointermove", (event) => {
		if (!drag) return;
		const rect = map.getBoundingClientRect();
		const scale = Math.min(rect.width / 840, rect.height / 480);
		pan = [drag[2] + (event.clientX - drag[0]) / scale, drag[3] + (event.clientY - drag[1]) / scale];
		camera();
	});
	for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) map.addEventListener(name, () => { drag = null; map.classList.remove("is-dragging"); });
	map.addEventListener("keydown", (event) => {
		const moves = { ArrowLeft: [40, 0], ArrowRight: [-40, 0], ArrowUp: [0, 40], ArrowDown: [0, -40] };
		if (moves[event.key]) pan = pan.map((value, axis) => value + moves[event.key][axis]);
		else if (event.key === "+" || event.key === "=") zoom = Math.min(3, zoom + .5);
		else if (event.key === "-") zoom = Math.max(1, zoom - .5);
		else if (event.key === "Home") { zoom = 1; pan = [0, 0]; }
		else return;
		event.preventDefault();
		draw();
	});
	new ResizeObserver(draw).observe(map);
	element("route-progress").addEventListener("input", (event) => { stop(); fraction = Number(event.target.value) / 1000; drawRoute(); });
	element("play-route").addEventListener("click", () => {
		if (playing) { stop(); return; }
		if (reducedMotion.matches) { fraction = 1; drawRoute(); element("map-caption").textContent = "Movimento reduzido ativado. Use o controle Percurso para explorar o caminho sem animação."; return; }
		if (fraction >= 1) fraction = 0;
		playing = true;
		element("play-route").textContent = "Ⅱ Pausar percurso";
		element("play-route").setAttribute("aria-pressed", "true");
		let previous = performance.now();
		const tick = (now) => {
			fraction = Math.min(1, fraction + (now - previous) / 8000);
			previous = now;
			drawRoute();
			if (fraction >= 1) stop();
			else frame = requestAnimationFrame(tick);
		};
		frame = requestAnimationFrame(tick);
	});
	document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); });
	reducedMotion.addEventListener("change", stop);
	element("result-filter").addEventListener("change", renderTable);
	element("case-search").addEventListener("input", renderTable);
	element("result-rows").addEventListener("click", (event) => {
		const button = event.target.closest("[data-open-case]");
		if (!button) return;
		selectCase(Number(button.dataset.openCase));
		element("explorar").scrollIntoView({ behavior: reducedMotion.matches ? "instant" : "smooth" });
		element("case-select").focus({ preventScroll: true });
	});
	element("download-case").addEventListener("click", () => download(`tpp-siicusp-caso-${row.case}.json`, JSON.stringify({ provenance: data.provenance, config: data.config, result: row }, null, 2), "application/json"));
	element("download-results").addEventListener("click", () => {
		const header = ["case", "polygons", "seconds", "lower_bound", "upper_bound", "relative_gap", "exact", "termination", "valid", "sha256"];
		const csv = [header.join(","), ...data.rows.map((item) => header.map((key) => key === "relative_gap" ? gapRatio(item) : item[key]).join(","))].join("\n");
		download("siicusp34-resultados-60-casos.csv", csv, "text/csv;charset=utf-8");
	});
	const requested = new URLSearchParams(window.location.search).get("caso");
	selectCase(requested !== null && /^\d+$/.test(requested) ? Number(requested) : 2, false) || selectCase(2, false);
	renderTable();
}

try { initialize(); }
catch (error) {
	element("event-error").hidden = false;
	element("event-error").textContent = "Não foi possível iniciar os controles interativos. Os resultados e a visualização estática continuam disponíveis. Recarregue a página para tentar novamente.";
	console.error(error);
}
