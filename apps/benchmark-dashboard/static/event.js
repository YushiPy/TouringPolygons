import { convexHull, endpointOffset, pathPrefix, projectedCase, regionColors, sortRows, sortGroupedRows, toggleChoice, toggleOrderRegion, playbackDuration } from "./event-geometry.js?v=20260910-7";

const element = (id) => document.getElementById(id);
const number = (value, digits = 4) => value.toLocaleString("pt-BR", { maximumFractionDigits: digits });
const coordinates = (points) => points.map((point) => point.join(",")).join(" ");
const caseLabel = (id) => String(id + 1).padStart(id + 1 >= 100 ? 3 : 2, "0");
const visitorRows = (rows, query) => rows.filter(row => !query.trim() || caseLabel(row.case).includes(query.trim()));
const titles = { 2: "Quatro regiões, um caminho", 9: "Quarenta regiões, ordem livre", 55: "Um caminho, uma prova em aberto" };
const MAX_ZOOM = 8;

function traceNumber(value, digits = 4) {
	return Number.isFinite(Number(value)) ? number(Number(value), digits) : "—";
}

function traceLabel(order, original) {
	const rank = Array.isArray(order) ? order.indexOf(Number(original)) : -1;
	return rank >= 0 ? String(rank + 1) : String(Number(original) + 1);
}

function traceLabels(order, sequence) {
	return Array.isArray(sequence) && sequence.length ? sequence.map((index) => traceLabel(order, index)).join(" → ") : "∅";
}

function traceEscape(value) {
	return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

function animateChallengeRoute(map) {
	const route = map.querySelector(".challenge-solution-route");
	if (!route || !route.animate || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
	const length = route.getTotalLength();
	route.style.strokeDasharray = `${length} ${length}`;
	route.style.strokeDashoffset = String(length);
	const animation = route.animate([{ strokeDashoffset: String(length) }, { strokeDashoffset: "0" }], {
		duration: 2000,
		easing: "cubic-bezier(.3,.65,.25,1)",
		fill: "forwards",
	});
	animation.finished.then(() => {
		route.style.strokeDasharray = "none";
		route.style.strokeDashoffset = "0";
		animation.cancel();
	}).catch(() => {});
}

function challengeComparison(chosenLabel, chosenLength, bestLabel, bestLength, choiceLabel, chosenChoice, bestChoice) {
	return `<table class="challenge-comparison"><thead><tr><td></td><th><i class="comparison-line chosen" aria-hidden="true"></i>${chosenLabel}</th><th><i class="comparison-line reference" aria-hidden="true"></i>${bestLabel}</th></tr></thead><tbody><tr><th scope="row">Comprimento</th><td>${number(chosenLength, 2)}</td><td>${number(bestLength, 2)}</td></tr><tr><th scope="row">${choiceLabel}</th><td>${chosenChoice}</td><td>${bestChoice}</td></tr></tbody></table><p class="comparison-note">Comprimentos em unidades deste exemplo.</p>`;
}

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
	const tracePayload = JSON.parse(element("trace-data")?.textContent || '{"schema_version":1,"cases":{}}');
	const traces = tracePayload.schema_version === 1 ? (tracePayload.cases || {}) : {};
	if (data.schema_version !== 1 || !data.rows.length) throw new Error("Dados da demonstração indisponíveis.");
	let row = data.rows.find((item) => item.case === 2);
	let projected, fraction = 1, zoom = 1, frame = 0, playing = false;
	let traceEvents = [], traceIndex = 0, traceTimer = null;
	let pan = [0, 0], drag = null;
	const speeds = [.25, .5, 1, 1.5, 2, 3, 4];
	let speedIndex = 2;
	let resultGroup = data.corpus === "german" ? false : null;
	let showAllResults = false;
	const sorting = { picker: { key: "case", descending: false }, result: { key: "case", descending: false } };
	const enabled = (id) => element(id).getAttribute("aria-pressed") === "true";
	const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
	const mapContent = element("map-content");
	const map = element("route-map");
	const traceMapContent = element("trace-map-content");
	const hasTraceUI = Boolean(traceMapContent);

	function stopTrace() {
		if (!hasTraceUI) return;
		if (traceTimer !== null) window.clearInterval(traceTimer);
		traceTimer = null;
		element("trace-play").textContent = "▶ Reproduzir";
		element("trace-play").setAttribute("aria-pressed", "false");
	}

	function traceCurrentPath(event) {
		for (let index = traceIndex; index >= 0; index -= 1) {
			if (Array.isArray(traceEvents[index]?.path) && traceEvents[index].path.length) return traceEvents[index].path;
		}
		return row.path;
	}

	function traceSequence(event) {
		if (Array.isArray(event?.order) && event.order.length) return event.order;
		return Array.isArray(event?.sequence) ? event.sequence : [];
	}

	function traceEventCopy(event, trace) {
		const order = trace.optimal_order || row.order || [];
		const selected = traceLabels(order, traceSequence(event));
		const polygon = event.polygon === undefined ? null : traceLabel(order, event.polygon);
		const kind = event.kind;
		if (kind === "heuristic_start") return ["A heurística começa", `O caminho inicial parte de S e procura uma ordem promissora de forma gulosa (${event.source === "reverse" ? "sentido reverso" : "sentido direto"}).`];
		if (kind === "heuristic_greedy_step") return ["Escolha gulosa", `A heurística acrescenta a região ${polygon}; a sequência parcial agora é ${selected}.`];
		if (kind === "heuristic_greedy_complete") return ["Ordem gulosa completa", `Todas as regiões foram inseridas. A ordem candidata é ${selected}.`];
		if (kind === "heuristic_2opt") return ["Refino 2-opt", event.reason === "changed" ? `A heurística trocou trechos da ordem e obteve ${selected}.` : "Nenhuma troca 2-opt melhorou a ordem; o refino estabilizou."];
		if (kind === "heuristic_contact_pass") return ["Refino geométrico", `O caminho da ordem ${selected} foi refinado nos pontos de contato (passagem ${Number(event.pass) + 1}).`];
		if (kind === "incumbent") return ["Novo incumbente", `Foi encontrado um caminho viável de comprimento ${traceNumber(event.length, 4)} para a ordem ${selected}.`];
		if (kind === "root") return ["Relaxação na raiz", "O solver ignora temporariamente os buracos e as regiões ainda não escolhidas para obter um limite inferior global."];
		if (kind === "expand") return ["Expande um nó", `A busca examina a ordem parcial ${selected} e calcula como ela pode ser completada.`];
		if (kind === "oracle") return ["Cálculo geométrico", `A relaxação convexa para ${selected} produz limite inferior ${traceNumber(event.lower_bound, 4)}${event.source === "refinement" ? "; a decomposição também foi refinada" : "."}`];
		if (kind === "branch") return ["Branching", `A árvore escolhe a região ${polygon} e cria alternativas de inserção${event.reason === "decomposition" ? " ou de peça convexa" : " na ordem"}.`];
		if (kind === "child") return [event.pruned ? "Filho podado" : "Filho enfileirado", event.piece === undefined ? (event.pruned ? `A sequência ${selected} tem limite ${traceNumber(event.lower_bound, 4)}, que já não pode melhorar o incumbente.` : `A sequência ${selected} permanece candidata e entra na fila de busca.`) : `A região ${polygon} foi refinada na peça convexa ${Number(event.piece) + 1}; ${event.pruned ? "o filho é podado" : "o filho entra na fila"}.`];
		if (kind === "prune") return ["Nó podado", event.reason === "incumbent" ? "O caminho encontrado já é tão bom quanto o incumbente." : "O limite inferior excede o melhor caminho conhecido."];
		if (kind === "complete") return ["Busca concluída", `A melhor ordem registrada é ${traceLabels(order, trace.optimal_order)}; comprimento ${traceNumber(event.length ?? trace.summary?.upper_bound, 4)}.`];
		return ["Passo da busca", "O solver atualizou o estado da busca."];
	}

	function renderTraceTree(trace) {
		const visibleKinds = new Set(["root", "expand", "branch", "child", "prune", "complete"]);
		const events = traceEvents.slice(0, traceIndex + 1).filter((event) => visibleKinds.has(event.kind));
		const visible = events.length > 42 ? events.slice(-42) : events;
		element("trace-tree").innerHTML = visible.length ? visible.map((event) => {
			const sequence = traceLabels(trace.optimal_order || row.order, event.sequence || []);
			const label = event.kind === "root" ? "Raiz" : event.kind === "expand" ? `Expande ${sequence}` : event.kind === "branch" ? `Branching em ${traceLabel(trace.optimal_order || row.order, event.polygon)}` : event.kind === "child" ? `${event.pruned ? "Poda" : "Fila"}: ${sequence}` : event.kind === "prune" ? `Poda: ${sequence}` : "Certificado";
			const detail = event.kind === "child" && Number.isFinite(Number(event.lower_bound)) ? `LB ${traceNumber(event.lower_bound, 3)}` : event.kind === "complete" ? `UB ${traceNumber(event.upper_bound ?? trace.summary?.upper_bound, 3)}` : "";
			const indent = Math.min(Array.isArray(event.sequence) ? event.sequence.length : 0, 8);
			return `<div class="trace-tree-item ${event.pruned ? "is-pruned" : ""} ${event.kind === "complete" ? "is-complete" : ""}" style="--trace-depth:${indent}"><span>${traceEscape(label)}</span><small>${traceEscape(detail)}</small></div>`;
		}).join("") : '<p class="trace-tree-empty">A árvore aparecerá quando a busca começar.</p>';
	}

	function drawTrace() {
		if (!hasTraceUI) return;
		const trace = traces[String(row.case)];
		if (!trace || !traceEvents.length) {
			traceMapContent.innerHTML = "";
			element("trace-progress").textContent = "Nenhuma simulação carregada para este caso.";
			element("trace-step-title").textContent = "Escolha uma instância didática";
			element("trace-step-text").textContent = "Os casos 03, 10 e 56 têm uma reprodução compacta da execução armazenada para esta demonstração.";
			element("trace-kind").textContent = "PASSO ATUAL";
			element("trace-sequence").textContent = "—";
			element("trace-lower-bound").textContent = "—";
			element("trace-upper-bound").textContent = "—";
			element("trace-previous").disabled = true;
			element("trace-next").disabled = true;
			element("trace-play").disabled = true;
			element("trace-tree").innerHTML = '<p class="trace-tree-empty">Selecione um caso com trace disponível.</p>';
			return;
		}
		const event = traceEvents[traceIndex];
		const order = trace.optimal_order || row.order || [];
		const currentSequence = traceSequence(event);
		const selected = new Set(currentSequence.map(Number));
		const projection = projectedCase(row);
		const start = projection.project(row.geometry.start), target = projection.project(row.geometry.target);
		const currentPath = traceCurrentPath(event).map(projection.project);
		const finalEvent = [...traceEvents].reverse().find((candidate) => candidate.kind === "complete" && Array.isArray(candidate.path) && candidate.path.length);
		const finalPath = (finalEvent?.path || row.path).map(projection.project);
		const showLabels = row.polygons <= 15 || selected.size > 0;
		const pieceIndex = Number.isInteger(Number(event.piece)) ? Number(event.piece) : -1;
		const branchPolygon = Number.isInteger(Number(event.polygon)) ? Number(event.polygon) : -1;
		const piece = branchPolygon >= 0 && pieceIndex >= 0 ? row.visualization?.decomposition?.[branchPolygon]?.[pieceIndex] : null;
		traceMapContent.innerHTML = `${row.geometry.polygons.map((polygon, index) => `<polygon class="trace-region ${selected.has(index) ? "trace-selected" : ""}" points="${coordinates(polygon.map(projection.project))}"><title>Região ${traceLabel(order, index)}</title></polygon>`).join("")}
			${[...selected].map((index) => `<polygon class="trace-hull" points="${coordinates(convexHull(row.geometry.polygons[index]).map(projection.project))}"/>`).join("")}
			${piece ? `<polygon class="trace-piece" points="${coordinates(piece.map(projection.project))}"/>` : ""}
			<polyline class="trace-final-route" points="${coordinates(finalPath)}"/><polyline class="trace-current-route" points="${coordinates(currentPath)}"/>
			${showLabels ? projection.polygons.map((polygon, index) => { const center = polygon.reduce((sum, point) => [sum[0] + point[0] / polygon.length, sum[1] + point[1] / polygon.length], [0, 0]); return `<text class="trace-region-label ${selected.has(index) ? "trace-label-selected" : ""}" x="${center[0]}" y="${center[1]}" text-anchor="middle">${traceLabel(order, index)}</text>`; }).join("") : ""}
			<circle class="trace-endpoint" cx="${start[0]}" cy="${start[1]}" r="6"/><text class="trace-endpoint-label" x="${start[0] + 13}" y="${start[1] + 4}">S</text><circle class="trace-endpoint trace-target" cx="${target[0]}" cy="${target[1]}" r="6"/><text class="trace-endpoint-label" x="${target[0] + 13}" y="${target[1] + 4}">T</text>`;
		const [title, text] = traceEventCopy(event, trace);
		element("trace-kind").textContent = event.kind.replaceAll("_", " ").toUpperCase();
		element("trace-step-title").textContent = title;
		element("trace-step-text").textContent = text;
		element("trace-sequence").textContent = traceLabels(order, currentSequence);
		element("trace-lower-bound").textContent = traceNumber(event.lower_bound, 4);
		element("trace-upper-bound").textContent = traceNumber(event.upper_bound, 4);
		element("trace-progress").textContent = `Passo ${traceIndex + 1} de ${traceEvents.length}${trace.omitted_events ? ` · ${trace.omitted_events.toLocaleString("pt-BR")} eventos omitidos` : ""}`;
		element("trace-previous").disabled = traceIndex === 0;
		element("trace-next").disabled = traceIndex === traceEvents.length - 1;
		element("trace-play").disabled = traceEvents.length < 2;
		element("trace-map-caption").textContent = `Regiões destacadas pertencem à sequência parcial ${traceLabels(order, currentSequence)}. A linha clara é o caminho final registrado; a laranja mostra o caminho disponível neste passo.`;
		renderTraceTree(trace);
	}

	function selectTrace() {
		if (!hasTraceUI) return;
		stopTrace();
		const trace = traces[String(row.case)];
		traceEvents = trace?.events || [];
		traceIndex = 0;
		drawTrace();
	}

	function advanceTrace(delta) {
		if (!traceEvents.length) return;
		traceIndex = Math.max(0, Math.min(traceEvents.length - 1, traceIndex + delta));
		drawTrace();
	}

	function camera() {
		pan = [Math.max(-420 * (zoom - 1), Math.min(420 * (zoom - 1), pan[0])), Math.max(-240 * (zoom - 1), Math.min(240 * (zoom - 1), pan[1]))];
		mapContent.setAttribute("transform", `translate(${420 + pan[0]} ${240 + pan[1]}) scale(${zoom}) translate(-420 -240)`);
		map.classList.toggle("is-zoomed", zoom > 1);
		map.style.touchAction = zoom > 1 ? "none" : "pan-y";
		element("zoom-out").disabled = zoom <= 1;
		element("zoom-in").disabled = zoom >= MAX_ZOOM;
		element("fit-view").disabled = zoom <= 1;
		element("map-navigation").textContent = zoom > 1 ? `Zoom ${number(zoom, 1)}×. Arraste para mover ou toque em Redefinir para voltar à rolagem da página.` : "Amplie para mover o desenho. Na visualização ajustada, arraste para continuar rolando a página.";
	}

	function stop() {
		cancelAnimationFrame(frame);
		playing = false;
		element("play-route").textContent = "▶ Veja o caminho";
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
		element("route-progress").style.setProperty("--progress", `${fraction * 100}%`);
		mapContent.querySelectorAll(".region").forEach((polygon) => {
			const index = Number(polygon.dataset.region);
			const contact = row.visualization.contacts[index];
			const visited = contact !== null && fraction + 1e-12 >= contact.fraction;
			const colors = regionColors(row.order.indexOf(index), row.polygons, visited);
			polygon.style.fill = colors.fill;
			polygon.style.stroke = colors.stroke;
			polygon.classList.toggle("visited", visited);
		});
		mapContent.querySelectorAll(".visit-contact").forEach((point) => {
			const contact = row.visualization.contacts[Number(point.dataset.region)];
			point.classList.toggle("reached", fraction + 1e-12 >= contact.fraction);
		});
	}

	function draw() {
		projected = projectedCase(row);
		const start = projected.project(row.geometry.start), target = projected.project(row.geometry.target);
		const labels = enabled("show-labels");
		const hulls = enabled("show-hulls");
		const decomposition = enabled("show-decomposition");
		mapContent.innerHTML = `${hulls ? row.geometry.polygons.map((polygon) => `<polygon class="hull" points="${coordinates(convexHull(polygon).map(projected.project))}"/>`).join("") : ""}
			${projected.polygons.map((polygon, index) => `<polygon class="region" data-region="${index}" points="${coordinates(polygon)}" style="fill:${regionColors(row.order.indexOf(index), row.polygons, fraction + 1e-12 >= (row.visualization.contacts[index]?.fraction ?? Infinity)).fill};stroke:${regionColors(row.order.indexOf(index), row.polygons, fraction + 1e-12 >= (row.visualization.contacts[index]?.fraction ?? Infinity)).stroke}"><title>Região ${index + 1}; ${row.order.indexOf(index) + 1}ª na ordem exportada</title></polygon>`).join("")}
			${decomposition ? row.visualization.decomposition.flatMap((pieces) => pieces.map((piece) => `<polygon class="convex-piece" points="${coordinates(piece.map(projected.project))}"/>`)).join("") : ""}
			<polyline class="route-ghost" points="${coordinates(projected.path)}"/><polyline id="animated-route" class="route-line"/>
			${labels ? projected.polygons.map((polygon, index) => {
				const center = polygon.reduce((sum, point) => [sum[0] + point[0] / polygon.length, sum[1] + point[1] / polygon.length], [0, 0]);
				return `<text class="region-label" x="${center[0]}" y="${center[1]}" text-anchor="middle">${row.order.indexOf(index) + 1}</text>`;
			}).join("") : ""}
			<circle class="endpoint" cx="${start[0]}" cy="${start[1]}" r="7"/><text class="endpoint-label" x="${start[0] + 14}" y="${start[1] + 5}">S</text>
			<circle class="endpoint target" cx="${target[0]}" cy="${target[1]}" r="7"/><text class="endpoint-label" x="${target[0] + 14}" y="${target[1] + 5}">T</text><circle id="traveler" class="traveler" r="6"/>
			${enabled("show-contacts") ? row.visualization.contacts.map((contact, index) => {
				if (!contact) return "";
				const point = projected.project(contact.point);
				return `<circle class="visit-contact" data-region="${index}" cx="${point[0]}" cy="${point[1]}" r="4"><title>${row.order.indexOf(index) + 1}ª visita · região original ${index + 1}</title></circle>`;
			}).join("") : ""}`;
		const rect = map.getBoundingClientRect();
		const textScale = 1 / Math.max(.1, Math.min(rect.width / 840, rect.height / 480)) / zoom;
		mapContent.querySelectorAll(".region-label").forEach((label) => { label.style.fontSize = `${13 * textScale}px`; });
		mapContent.querySelectorAll(".endpoint-label").forEach((label, index) => {
			const point = endpointOffset(projected.path, Boolean(index), 22 * textScale);
			label.setAttribute("text-anchor", "middle");
			label.setAttribute("dominant-baseline", "central");
			label.style.fontSize = `${16 * textScale}px`;
			label.setAttribute("x", point[0]);
			label.setAttribute("y", point[1]);
		});
		mapContent.querySelectorAll(".endpoint, .traveler").forEach((point) => point.setAttribute("r", 6 * textScale));
		mapContent.querySelectorAll(".visit-contact").forEach((point) => point.setAttribute("r", 4 * textScale));
		camera();
		document.querySelectorAll("[data-layer-note]").forEach((item) => item.classList.toggle("active", enabled(item.dataset.layerNote)));
		element("map-caption").textContent = "Os números mostram a ordem da primeira visita. Basta tocar a borda ou atravessar a região; seus centros não são pontos obrigatórios.";
		element("map-title").textContent = `Caso ${caseLabel(row.case)}: caminho por ${row.polygons} regiões; ${row.exact ? "solução exata certificada" : "limite de tempo"}.`;
		drawRoute();
	}

	function selectCase(index, updateURL = true) {
		const selected = data.rows.find((item) => item.case === index);
		if (!selected) return false;
		stop();
		stopTrace();
		row = selected;
		fraction = 1;
		zoom = 1;
		pan = [0, 0];
		element("show-labels").setAttribute("aria-pressed", String(row.polygons <= 15));
		element("case-select").value = row.case;
		element("case-picker-value").textContent = `Caso ${caseLabel(row.case)} · ${row.polygons} regiões · ${row.exact ? "solução exata" : "limite de tempo"}`;
		document.querySelectorAll(".example").forEach((button) => {
			const active = Number(button.dataset.case) === row.case;
			button.classList.toggle("active", active);
			button.setAttribute("aria-pressed", String(active));
			if (active) button.setAttribute("aria-current", "true");
			else button.removeAttribute("aria-current");
		});
		element("drawing-title").textContent = data.corpus === "german" ? `${row.polygons} regiões, caminho registrado` : titles[row.case] || `${row.polygons} regiões, extremos fixos`;
		element("speed-value").title = `A 1×, este caso leva ${number(playbackDuration(row.polygons) / 1000, 1)} s para percorrer o caminho.`;
		element("case-id").textContent = `Caso ${caseLabel(row.case)}`;
		element("outcome-badge").className = `status ${row.exact ? "certified" : "limited"}`;
		element("outcome-badge").textContent = row.exact ? "✓ Solução exata" : "◷ Encerrado por tempo";
		element("outcome-title").textContent = row.exact ? "Ótimo certificado" : "Caminho encontrado; certificação pendente";
		element("outcome-explanation").textContent = row.exact ? "A busca terminou e o solver certificou este caminho como ótimo." : "O caminho visita todas as regiões. O tempo terminou antes da certificação exata.";
		element("case-length").textContent = number(row.length ?? row.validation?.recomputed_length, 2);
		element("case-time").textContent = row.seconds < .001 ? "< 0,001 s" : `${number(row.seconds, 3)} s`;
		element("case-regions").textContent = `${row.polygons} / ${row.polygons}`;
		element("case-certification").textContent = row.exact ? "Exata" : "Pendente";
		element("case-certification-note").textContent = row.exact ? "busca concluída" : "limite de tempo atingido";
		if (updateURL && window.location.protocol !== "file:") {
			const url = new URL(window.location.href);
			url.searchParams.set("caso", row.case);
			window.history.replaceState(null, "", url);
		}
		draw();
		selectTrace();
		return true;
	}

	function renderTable() {
		const rows = sortGroupedRows(data.rows, sorting.result.key, sorting.result.descending, resultGroup);
		const visible = showAllResults ? rows : rows.slice(0, 8);
		element("result-count").textContent = "Selecione um caso para ver o caminho.";
		const toggle = rows.length > 8 ? `<tr class="result-toggle-row"><td colspan="5"><button type="button" data-toggle-results aria-expanded="${showAllResults}">${showAllResults ? "Mostrar somente os 8 destaques ↑" : `Ver todos os ${rows.length} resultados ↓`}</button></td></tr>` : "";
		element("result-rows").innerHTML = visible.map((item) => `<tr><th scope="row"><span class="mobile-case-label">Caso </span>${caseLabel(item.case)}</th><td data-label="Regiões">${item.polygons}</td><td data-label="Tempo">${item.seconds < .001 ? "< 0,001 s" : `${number(item.seconds, 3)} s`}</td><td class="result-status"><span class="status ${item.exact ? "certified" : "limited"}">${item.exact ? "✓ Solução exata" : "◷ Limite de tempo"}</span></td><td class="result-action"><button type="button" data-open-case="${item.case}" aria-label="Ver caminho do caso ${item.case + 1}">Ver caminho →</button></td></tr>`).join("") + toggle;
	}

	document.querySelectorAll("button:disabled, input:disabled, select:disabled").forEach((control) => { control.disabled = false; });
	function showMap() {
		element("route-map").scrollIntoView({ behavior: reducedMotion.matches ? "instant" : "smooth", block: "center" });
		element("play-route").focus({ preventScroll: true });
	}
	document.querySelectorAll(".example").forEach((button) => button.addEventListener("click", () => {
		if (Number(button.dataset.case) === row.case) return;
		selectCase(Number(button.dataset.case));
		showMap();
	}));
	element("case-select").addEventListener("change", (event) => { selectCase(Number(event.target.value)); showMap(); });
	element("show-labels").addEventListener("click", () => { element("show-labels").setAttribute("aria-pressed", String(!enabled("show-labels"))); draw(); });
	element("show-hulls").addEventListener("click", () => { element("show-hulls").setAttribute("aria-pressed", String(!enabled("show-hulls"))); draw(); });
	element("show-contacts").addEventListener("click", () => { element("show-contacts").setAttribute("aria-pressed", String(!enabled("show-contacts"))); draw(); });
	element("show-decomposition").addEventListener("click", () => { element("show-decomposition").setAttribute("aria-pressed", String(!enabled("show-decomposition"))); draw(); });
	element("fit-view").addEventListener("click", () => { zoom = 1; pan = [0, 0]; draw(); });
	element("zoom-out").addEventListener("click", () => zoomAt(zoom / 1.5, [0, 0]));
	element("zoom-in").addEventListener("click", () => zoomAt(zoom * 1.5, [0, 0]));
	const pointers = new Map();
	let gesture = null;
	function localPoint(event) {
		const box = map.getBoundingClientRect();
		const scale = Math.min(box.width / 840, box.height / 480);
		return [(event.clientX - box.left - box.width / 2) / scale, (event.clientY - box.top - box.height / 2) / scale];
	}
	function zoomAt(next, point) {
		const previous = zoom;
		zoom = Math.max(1, Math.min(MAX_ZOOM, next));
		pan = point.map((value, axis) => value - (value - pan[axis]) * zoom / previous);
		draw();
	}
	let trackpadGesture = null;
	map.addEventListener("wheel", (event) => {
		if (trackpadGesture) return;
		const units = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 480 : 1;
		if (!event.ctrlKey && zoom <= 1 && event.deltaY * units > 0) return;
		event.preventDefault();
		zoomAt(zoom * Math.exp(-event.deltaY * units * (event.ctrlKey ? .01 : .002)), localPoint(event));
	}, { passive: false });
	map.addEventListener("gesturestart", (event) => { event.preventDefault(); trackpadGesture = { zoom, scale: 1 }; }, { passive: false });
	map.addEventListener("gesturechange", (event) => {
		event.preventDefault();
		if (!trackpadGesture || !Number.isFinite(event.scale)) return;
		zoomAt(trackpadGesture.zoom * event.scale, localPoint(event));
	}, { passive: false });
	map.addEventListener("gestureend", (event) => { event.preventDefault(); trackpadGesture = null; }, { passive: false });
	function resetGesture() {
		const points = [...pointers.values()];
		if (points.length >= 2) {
			const center = points[0].map((v, i) => (v + points[1][i]) / 2);
			gesture = { distance: Math.hypot(...points[0].map((v, i) => v - points[1][i])), zoom, center, pan: [...pan] };
		} else { gesture = null; drag = points.length ? [...points[0], ...pan] : null; }
	}
	map.addEventListener("pointerdown", (event) => {
		if (event.button !== 0) return;
		if (event.pointerType === "touch" && zoom <= 1 && event.isPrimary) return;
		map.focus({ preventScroll: true });
		pointers.set(event.pointerId, localPoint(event));
		map.setPointerCapture(event.pointerId);
		resetGesture();
	});
	map.addEventListener("pointermove", (event) => {
		if (!pointers.has(event.pointerId)) return;
		const point = localPoint(event);
		pointers.set(event.pointerId, point);
		if (gesture && pointers.size >= 2) {
			const points = [...pointers.values()];
			const distance = Math.hypot(...points[0].map((v, i) => v - points[1][i]));
			zoom = Math.max(1, Math.min(MAX_ZOOM, gesture.zoom * distance / Math.max(gesture.distance, 1e-6)));
			pan = points[0].map((v, i) => (v + points[1][i]) / 2 - (gesture.center[i] - gesture.pan[i]) * zoom / gesture.zoom);
			draw();
		} else if (drag) { pan = point.map((v, i) => drag[i + 2] + v - drag[i]); camera(); }
	});
	for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) map.addEventListener(name, (event) => {
		pointers.delete(event.pointerId); resetGesture();
	});
	map.addEventListener("keydown", (event) => {
		const moves = { ArrowLeft: [40, 0], ArrowRight: [-40, 0], ArrowUp: [0, 40], ArrowDown: [0, -40] };
		if (moves[event.key]) pan = pan.map((value, axis) => value + moves[event.key][axis]);
		else if (event.key === "+" || event.key === "=") zoom = Math.min(MAX_ZOOM, zoom + .5);
		else if (event.key === "-" || event.key === "−") zoom = Math.max(1, zoom - .5);
		else if (event.key === "Home") { zoom = 1; pan = [0, 0]; }
		else return;
		event.preventDefault();
		draw();
	});
	new ResizeObserver(draw).observe(map);
	element("route-progress").addEventListener("input", (event) => { stop(); fraction = Number(event.target.value) / 1000; drawRoute(); });
	element("play-route").addEventListener("click", () => {
		if (playing) { stop(); return; }
		if (reducedMotion.matches) { fraction = 1; drawRoute(); element("map-caption").textContent = "Movimento reduzido ativado. Use o controle Progresso para explorar o caminho sem animação."; return; }
		if (fraction >= 1) fraction = 0;
		playing = true;
		element("play-route").textContent = "Ⅱ Pausar percurso";
		element("play-route").setAttribute("aria-pressed", "true");
		let previous = performance.now();
		const tick = (now) => {
			fraction = Math.min(1, fraction + (now - previous) * speeds[speedIndex] / playbackDuration(row.polygons));
			previous = now;
			drawRoute();
			if (fraction >= 1) stop();
			else frame = requestAnimationFrame(tick);
		};
		frame = requestAnimationFrame(tick);
	});
	if (hasTraceUI) {
		element("trace-previous").addEventListener("click", () => advanceTrace(-1));
		element("trace-next").addEventListener("click", () => advanceTrace(1));
		element("trace-play").addEventListener("click", () => {
			if (!traceEvents.length) return;
			if (traceTimer !== null) { stopTrace(); return; }
			if (traceIndex >= traceEvents.length - 1) traceIndex = 0;
			element("trace-play").textContent = "Ⅱ Pausar";
			element("trace-play").setAttribute("aria-pressed", "true");
			traceTimer = window.setInterval(() => {
				if (traceIndex >= traceEvents.length - 1) { stopTrace(); return; }
				advanceTrace(1);
			}, reducedMotion.matches ? 1400 : 850);
		});
	}
	document.addEventListener("visibilitychange", () => { if (document.hidden) { stop(); stopTrace(); } });
	reducedMotion.addEventListener("change", stop);
	let returnContext = null;
	element("result-rows").addEventListener("click", (event) => {
		const toggle = event.target.closest("[data-toggle-results]");
		if (toggle) {
			showAllResults = !showAllResults;
			renderTable();
			element("result-rows").querySelector("[data-toggle-results]")?.focus({ preventScroll: true });
			return;
		}
		const button = event.target.closest("[data-open-case]");
		if (!button) return;
		returnContext = { top: window.scrollY, control: button };
		selectCase(Number(button.dataset.openCase));
		element("context-return").hidden = false;
		showMap();
	});
	element("context-return").addEventListener("click", () => {
		if (!returnContext) return;
		const destination = returnContext;
		returnContext = null;
		element("context-return").hidden = true;
		const start = window.scrollY;
		const finish = () => destination.control.focus({ preventScroll: true });
		if (reducedMotion.matches || Math.abs(start - destination.top) < 2) {
			window.scrollTo({ top: destination.top, behavior: "instant" });
			finish();
			return;
		}
		window.scrollTo({ top: start, behavior: "instant" });
		const started = performance.now();
		const animate = (now) => {
			const progress = Math.min(1, (now - started) / 260);
			const eased = 1 - (1 - progress) ** 3;
			window.scrollTo(0, start + (destination.top - start) * eased);
			if (progress < 1) requestAnimationFrame(animate);
			else finish();
		};
		requestAnimationFrame(animate);
	});
	function setSpeed(index) {
		speedIndex = Math.max(0, Math.min(speeds.length - 1, index));
		element("speed-value").textContent = `${number(speeds[speedIndex], 2)}×`;
		element("speed-down").disabled = speedIndex === 0;
		element("speed-up").disabled = speedIndex === speeds.length - 1;
	}
	setSpeed(speedIndex);
	element("speed-down").addEventListener("click", () => setSpeed(speedIndex - 1));
	element("speed-up").addEventListener("click", () => setSpeed(speedIndex + 1));
	const picker = element("case-picker-dialog");
	function renderPicker() {
		const rows = sortRows(visitorRows(data.rows, element("picker-search").value), sorting.picker.key, sorting.picker.descending);
		element("picker-count").textContent = `${rows.length} casos disponíveis`;
		element("picker-options").innerHTML = rows.length ? rows.map((item) => `<button type="button" data-pick-case="${item.case}" aria-pressed="${item.case === row.case}"><span><strong>Caso ${caseLabel(item.case)}</strong><small>${item.polygons} regiões · ${number(item.seconds, 3)} s</small></span><span class="status ${item.exact ? "certified" : "limited"}">${item.exact ? "✓ Solução exata" : "◷ Limite de tempo"}</span></button>`).join("") : "<p>Nenhum caso encontrado. Experimente outro número.</p>";
	}
	element("case-picker-button").addEventListener("click", () => {
		element("picker-search").value = "";
		renderPicker();
		picker.showModal();
		element("case-picker-button").setAttribute("aria-expanded", "true");
		element("picker-close").focus();
	});
	element("picker-close").addEventListener("click", () => picker.close());
	picker.addEventListener("close", () => {
		element("case-picker-button").setAttribute("aria-expanded", "false");
		element("case-picker-button").focus({ preventScroll: true });
	});
	picker.addEventListener("click", (event) => { if (event.target === picker) {
		const box = picker.getBoundingClientRect();
		if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) picker.close();
	} });
	element("picker-search").addEventListener("input", renderPicker);
	element("picker-options").addEventListener("click", (event) => {
		const button = event.target.closest("[data-pick-case]");
		if (button) { selectCase(Number(button.dataset.pickCase)); picker.close(); showMap(); }
	});
	element("picker-options").addEventListener("keydown", (event) => {
		const buttons = [...element("picker-options").querySelectorAll("button")];
		const index = buttons.indexOf(document.activeElement);
		if (event.key === "ArrowDown" || event.key === "ArrowUp") {
			event.preventDefault();
			buttons[(index + (event.key === "ArrowDown" ? 1 : -1) + buttons.length) % buttons.length]?.focus();
		}
	});
	for (const prefix of ["picker"]) {
		const direction = element(`${prefix}-direction`);
		const group = direction.parentElement;
		const render = prefix === "picker" ? renderPicker : renderTable;
		group.querySelectorAll("[data-sort]").forEach((button) => button.addEventListener("click", () => {
			sorting[prefix].key = button.dataset.sort;
			group.querySelectorAll("[data-sort]").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
			render();
		}));
		direction.addEventListener("click", () => {
			sorting[prefix].descending = !sorting[prefix].descending;
			direction.setAttribute("aria-pressed", String(sorting[prefix].descending));
			direction.textContent = sorting[prefix].descending ? "↓ Decrescente" : "↑ Crescente";
			render();
		});
	}
	function updateResultSorting() {
		const names = { case: "caso", polygons: "regiões", seconds: "tempo" };
		document.querySelectorAll("[data-result-column]").forEach((header) => {
			const key = header.dataset.resultColumn;
			const active = key === "result" ? resultGroup !== null : key === sorting.result.key;
			const descending = key === "result" ? resultGroup : sorting.result.descending;
			const primary = resultGroup === null ? key === sorting.result.key : key === "result";
			header.setAttribute("aria-sort", active && primary ? (descending ? "descending" : "ascending") : "none");
			header.querySelector(".sort-arrow").textContent = active ? (descending ? " ↓" : " ↑") : "";
			header.classList.toggle("sort-active", active);
		});
		element("result-sort-description").textContent = `${resultGroup === null ? "Sem agrupamento; " : resultGroup ? "Limites de tempo primeiro; depois " : "Soluções exatas primeiro; depois "}${names[sorting.result.key]} em ordem ${sorting.result.descending ? "decrescente" : "crescente"}. Resultado alterna entre soluções exatas primeiro, limite de tempo primeiro e sem agrupamento.`;
		renderTable();
	}
	document.querySelectorAll("[data-result-sort]").forEach((button) => button.addEventListener("click", () => {
		const key = button.dataset.resultSort;
		if (key === "result") resultGroup = resultGroup === null ? false : resultGroup === false ? true : null;
		else {
			sorting.result.descending = sorting.result.key === key ? !sorting.result.descending : false;
			sorting.result.key = key;
		}
		updateResultSorting();
	}));
	updateResultSorting();
	element("case-picker").hidden = false;
	element("case-select").hidden = true;
	document.querySelector('label[for="case-select"]').setAttribute("for", "case-picker-button");
	element("download-case").addEventListener("click", () => {
		const result = { ...row };
		download(`tpp-caso-${row.case + 1}.json`, JSON.stringify({ case_number: row.case + 1, provenance: data.provenance, config: data.config, result }, null, 2), "application/json");
	});
	element("download-results").addEventListener("click", () => {
		const header = ["case_number", "case", "polygons", "seconds", "exact", "termination", "valid", "sha256"];
		const csv = [header.join(","), ...data.rows.map((item) => header.map((key) => key === "case_number" ? item.case + 1 : key === "valid" ? (item.validation?.valid ?? item.valid ?? "") : item[key]).join(","))].join("\n");
		download(`tpp-resultados-${data.rows.length}-casos.csv`, csv, "text/csv;charset=utf-8");
	});
	const requested = new URLSearchParams(window.location.search).get("caso");
	selectCase(requested !== null && /^\d+$/.test(requested) ? Number(requested) : 2, false) || selectCase(2, false);
	renderTable();
	if (element("challenge-data")) {
		initializeChallenge();
		initializePieceChallenge();
		initializePieceChallenge(true);
		initializeChallengeFlow();
	}
	if (element("references-dialog")) initializeReferences();
	initializeDisclosures();
	initializeGuide();
	if (element("toc-handle")) initializeContents();
}

function orderSketch(id, geometry, polygons) {
	if (!polygons.length) return "";
	const centers = polygons.map((polygon) => polygon.reduce((sum, point) => sum.map((value, axis) => value + point[axis] / polygon.length), [0, 0]));
	const points = [geometry.start, ...centers, geometry.target];
	return `<g class="order-sketch" aria-hidden="true"><defs><marker id="sketch-${id}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M1 1L8 5L1 9" fill="none" stroke="#c3d9e5" stroke-width="1.8"/></marker></defs>${points.slice(1).map((point, index) => `<path d="M${points[index].join(" ")}L${point.join(" ")}" marker-end="url(#sketch-${id})"/>`).join("")}</g>`;
}

function initializeChallenge() {
	const challenge = JSON.parse(element("challenge-data").textContent);
	const letters = ["A", "B", "C", "D"];
	const best = challenge.solutions[challenge.reference];
	let order = [], compared = false;
	const map = element("challenge-map");
	map.setAttribute("role", "group");
	function render() {
		const chosen = challenge.solutions.find((solution) => solution.order.join() === order.join());
		map.innerHTML = challenge.geometry.polygons.map((polygon, index) => {
			const x = (polygon[0][0] + polygon[2][0]) / 2, y = (polygon[0][1] + polygon[2][1]) / 2;
			const rank = order.indexOf(index);
			return `<g role="button" tabindex="0" data-challenge-region="${index}" aria-label="Região ${letters[index]}${rank >= 0 ? `, escolha ${rank + 1}` : ""}" aria-pressed="${rank >= 0}"><polygon points="${coordinates(polygon)}" class="challenge-region ${rank >= 0 ? "chosen" : ""}"/><text x="${x}" y="${y}" text-anchor="middle" dominant-baseline="central">${letters[index]}${rank >= 0 ? ` · ${rank + 1}` : ""}</text></g>`;
		}).join("") + (compared ? `<polyline class="challenge-reference" points="${coordinates(best.path)}"/><polyline class="route-line challenge-solution-route" points="${coordinates(chosen.path)}"/>` : orderSketch("order", challenge.geometry, order.map((index) => challenge.geometry.polygons[index]))) + '<circle cx="25" cy="130" r="5" fill="white"/><text x="17" y="153">S</text><circle cx="395" cy="130" r="5" fill="#ffad66"/><text x="389" y="153">T</text>';
		element("challenge-choices").innerHTML = letters.map((letter, index) => `<button type="button" data-challenge-region="${index}" aria-pressed="${order.includes(index)}">${letter}</button>`).join("");
		element("challenge-order").textContent = `Sua ordem: S → ${order.length ? order.map((index) => letters[index]).join(" → ") + " → " : ""}${order.length < 4 ? "… → " : ""}T`;
		element("challenge-guidance").textContent = order.length === 4 ? "Sequência completa. Compare agora os caminhos." : `Escolha mais ${4 - order.length} ${4 - order.length === 1 ? "região" : "regiões"} para comparar.`;
		element("challenge-compare").disabled = order.length !== 4 || compared;
		element("challenge-undo").disabled = !order.length;
		element("challenge-reset").disabled = !order.length;
		element("challenge-feedback").innerHTML = compared ? `<strong>${chosen.length - best.length < .00001 ? "Você encontrou uma das melhores ordens!" : `Seu caminho ficou ${number(100 * (chosen.length / best.length - 1), 1)}% mais longo.`}</strong>${challengeComparison("Sua escolha", chosen.length, "Melhor das 24", best.length, "Ordem", order.map((index) => letters[index]).join(" → "), best.order.map((index) => letters[index]).join(" → "))}` : "";
		if (compared) animateChallengeRoute(map);
	}
	function choose(event) {
		const control = event.target.closest("[data-challenge-region]");
		if (!control) return;
		const index = Number(control.dataset.challengeRegion);
		order = toggleOrderRegion(order, index);
		compared = false;
		element("challenge-next-one").hidden = true;
		const fromMap = map.contains(control);
		render();
		if (event.type === "keydown" || event.detail === 0) {
			const parent = fromMap ? map : element("challenge-choices");
			parent.querySelector(`[data-challenge-region="${index}"]`).focus({ preventScroll: true });
		}
	}
	map.addEventListener("click", choose);
	map.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(event); } });
	element("challenge-choices").addEventListener("click", choose);
	element("challenge-compare").addEventListener("click", () => { compared = true; render(); element("challenge-next-one").hidden = false; });
	element("challenge-undo").addEventListener("click", () => { order.pop(); compared = false; element("challenge-next-one").hidden = true; render(); });
	element("challenge-reset").addEventListener("click", () => { order = []; compared = false; element("challenge-next-one").hidden = true; render(); map.querySelector("[data-challenge-region]").focus({ preventScroll: true }); });
	render();
}

function initializePieceChallenge(combined = false) {
	const all = JSON.parse(element("challenge-data").textContent);
	const data = combined ? all.combined_challenge : all.piece_challenge;
	const prefix = combined ? "combined" : "piece";
	const ui = (suffix) => element(`${prefix}-${suffix}`);
	let order = [];
	const best = data.solutions[data.reference];
	const letters = combined ? ["A", "B", "C", "D"] : ["A", "B", "C"];
	let choices = letters.map(() => null), compared = false;
	const map = ui("map");
	function render() {
		const selected = data.solutions.find((item) => item.choices.every((piece, index) => piece === choices[index]) && (!combined || item.order.join() === order.join()));
		map.innerHTML = data.pieces.map((pieces, region) => {
			const rank = order.indexOf(region);
			const bounds = data.geometry.polygons[region].reduce((box, point) => ({
				minX: Math.min(box.minX, point[0]), maxX: Math.max(box.maxX, point[0]), minY: Math.min(box.minY, point[1]),
			}), { minX: Infinity, maxX: -Infinity, minY: Infinity });
			const regionName = combined && rank >= 0 ? `${letters[region]} · ${rank + 1}ª` : letters[region];
			const pieceMarkup = pieces.map((piece, index) => {
				const x = (piece[0][0] + piece[2][0]) / 2, y = (piece[0][1] + piece[2][1]) / 2;
				return `<g role="button" tabindex="0" data-piece="${index}" data-piece-region="${region}" aria-label="Região ${letters[region]}, peça ${index + 1}${rank >= 0 ? `, visita ${rank + 1}` : ""}" aria-pressed="${choices[region] === index}"><polygon class="challenge-region ${choices[region] === index ? "chosen" : ""}" points="${coordinates(piece)}"/><text x="${x}" y="${y}" dominant-baseline="central" text-anchor="middle">${index + 1}</text></g>`;
			}).join("");
			return `${pieceMarkup}<text class="challenge-region-name" x="${(bounds.minX + bounds.maxX) / 2}" y="${Math.max(18, bounds.minY - 10)}" text-anchor="middle">${regionName}</text>`;
		}).join("") + (compared ? `<polyline class="challenge-reference" points="${coordinates(best.path)}"/><polyline class="route-line challenge-solution-route" points="${coordinates(selected.path)}"/>` : orderSketch(prefix, data.geometry, combined ? order.map(index => data.pieces[index][choices[index]]) : choices.slice(0, choices.includes(null) ? choices.indexOf(null) : choices.length).map((piece, region) => data.pieces[region][piece]))) + `<circle cx="${data.geometry.start[0]}" cy="${data.geometry.start[1]}" r="5" fill="white"/><text x="${data.geometry.start[0] - 6}" y="${data.geometry.start[1] + 22}">S</text><circle cx="${data.geometry.target[0]}" cy="${data.geometry.target[1]}" r="5" fill="#ffad66"/><text x="${data.geometry.target[0] - 6}" y="${data.geometry.target[1] + 22}">T</text>`;
		ui("choices").innerHTML = letters.map((letter, region) => `<fieldset><legend>Região ${letter}</legend>${[0, 1, 2].map((piece) => `<button type="button" data-piece="${piece}" data-piece-region="${region}" aria-pressed="${choices[region] === piece}" aria-label="${letter}: peça ${piece + 1}">${piece + 1}</button>`).join("")}</fieldset>`).join("");
		ui("selection").textContent = choices.map((piece, region) => `${letters[region]}: ${piece === null ? "?" : `peça ${piece + 1}`}`).join(" · ");
		if (combined) ui("selection").textContent = `Sua sequência: S → ${order.map(index => `${letters[index]}${choices[index] + 1}`).join(" → ")}${order.length < letters.length ? " → …" : ""} → T`;
		const remaining = choices.filter((piece) => piece === null).length;
		ui("guidance").textContent = remaining ? `Faltam ${remaining} ${remaining === 1 ? "região" : "regiões"}.` : "Escolha completa. Compare agora os caminhos.";
		ui("compare").disabled = choices.includes(null) || compared;
		ui("reset").disabled = choices.every((piece) => piece === null);
		const chosenSequence = (combined ? order : letters.map((_, index) => index)).map(region => `${letters[region]}${choices[region] + 1}`).join(" → ");
		const bestSequence = (combined ? best.order : letters.map((_, index) => index)).map(region => `${letters[region]}${best.choices[region] + 1}`).join(" → ");
		const impact = combined ? '<aside class="challenge-impact"><strong>Achou difícil?</strong> Em um caso da pesquisa com mais de 2 × 10<sup>66</sup> possibilidades brutas, nosso algoritmo certificou o melhor caminho em 1,51 segundo. Mesmo que 10 bilhões de computadores testassem uma combinação por nanossegundo, levariam cerca de 7 × 10<sup>39</sup> anos, muito depois de o Sol se tornar uma gigante vermelha e possivelmente engolir a Terra <a href="#ref-sun">[5]</a>.</aside>' : "";
		ui("feedback").innerHTML = compared ? `<strong>${selected.length - best.length < .00001 ? "Você encontrou uma das melhores combinações!" : `Seu caminho ficou ${number(100 * (selected.length / best.length - 1), 1)}% mais longo.`}</strong>${challengeComparison("Sua escolha", selected.length, `Melhor das ${data.solutions.length.toLocaleString("pt-BR")}`, best.length, combined ? "Solução" : "Peças", chosenSequence, bestSequence)}${impact}` : "";
		if (compared) animateChallengeRoute(map);
	}
	function choose(event) {
		const control = event.target.closest("[data-piece]");
		if (!control) return;
		const region = Number(control.dataset.pieceRegion), piece = Number(control.dataset.piece);
		const parent = map.contains(control) ? map : ui("choices");
		if (combined) {
			if (choices[region] === piece) { choices[region] = null; order = order.filter(index => index !== region); }
			else { if (choices[region] === null) order.push(region); choices[region] = piece; }
		} else choices[region] = toggleChoice(choices[region], piece);
		compared = false;
		if (combined) element("finish-challenge").hidden = true;
		else element("challenge-next-two").hidden = true;
		render();
		if (event.type === "keydown" || event.detail === 0) parent.querySelector(`[data-piece-region="${region}"][data-piece="${piece}"]`).focus({ preventScroll: true });
	}
	map.addEventListener("click", choose);
	map.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(event); } });
	ui("choices").addEventListener("click", choose);
	ui("compare").addEventListener("click", () => {
		compared = true;
		render();
		if (combined) element("finish-challenge").hidden = false;
		else element("challenge-next-two").hidden = false;
	});
	ui("reset").addEventListener("click", () => {
		choices = letters.map(() => null);
		order = [];
		compared = false;
		if (combined) element("finish-challenge").hidden = true;
		else element("challenge-next-two").hidden = true;
		render();
		map.querySelector("[data-piece]").focus({ preventScroll: true });
	});
	render();
}

function initializeChallengeFlow() {
	const dialog = element("challenge-dialog");
	const steps = [...dialog.querySelectorAll("[data-challenge-step]")];
	const titles = ["Escolha a ordem das visitas", "Escolha uma peça em cada região", "Escolha a ordem e as peças"];
	let current = 0;
	function show(index) {
		current = Math.max(0, Math.min(steps.length - 1, index));
		steps.forEach((step, position) => { step.hidden = position !== current; });
		element("challenge-progress").textContent = `DESAFIO ${current + 1} DE ${steps.length}`;
		element("challenge-dialog-title").textContent = titles[current];
		dialog.querySelectorAll("[data-challenge-tab]").forEach((button, position) => {
			button.classList.toggle("visited", position <= current);
			button.classList.toggle("active", position === current);
			if (position === current) button.setAttribute("aria-current", "step");
			else button.removeAttribute("aria-current");
		});
		dialog.scrollTo({ top: 0, behavior: "instant" });
		const heading = steps[current].querySelector("h3");
		heading.setAttribute("tabindex", "-1");
		heading.focus({ preventScroll: true });
	}
	element("open-challenge").addEventListener("click", () => { dialog.showModal(); show(0); });
	element("close-challenge").addEventListener("click", () => dialog.close());
	element("challenge-next-one").addEventListener("click", () => show(1));
	element("challenge-next-two").addEventListener("click", () => show(2));
	dialog.querySelectorAll("[data-challenge-tab]").forEach((button) => button.addEventListener("click", () => show(Number(button.dataset.challengeTab))));
	dialog.querySelectorAll("[data-challenge-back]").forEach((button) => button.addEventListener("click", () => show(current - 1)));
	element("finish-challenge").addEventListener("click", () => dialog.close());
	dialog.addEventListener("click", (event) => {
		if (event.target !== dialog) return;
		const box = dialog.getBoundingClientRect();
		if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
	});
	dialog.addEventListener("close", () => element("open-challenge").focus({ preventScroll: true }));
}

function initializeReferences() {
	const dialog = element("references-dialog");
	let opener = null;
	document.addEventListener("click", (event) => {
		const trigger = event.target.closest('[data-open-references], a[href^="#ref-"]');
		if (!trigger || dialog.contains(trigger)) return;
		event.preventDefault();
		opener = trigger;
		if (element("mobile-toc")?.open) element("mobile-toc").close();
		if (!dialog.open) dialog.showModal();
		const selector = trigger.getAttribute("href");
		const target = selector?.startsWith("#ref-") ? dialog.querySelector(selector) : null;
		dialog.querySelectorAll("li.current-reference").forEach((item) => item.classList.remove("current-reference"));
		target?.classList.add("current-reference");
		requestAnimationFrame(() => target?.scrollIntoView({ behavior: "smooth", block: "center" }));
	});
	element("references-close").addEventListener("click", () => dialog.close());
	dialog.addEventListener("click", (event) => {
		if (event.target !== dialog) return;
		const box = dialog.getBoundingClientRect();
		if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
	});
	dialog.addEventListener("close", () => opener?.focus({ preventScroll: true }));
}

function initializeDisclosures() {
	const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
	document.querySelectorAll("details").forEach((details) => {
		const summary = details.querySelector(":scope > summary");
		if (!summary) return;
		const body = document.createElement("div");
		body.className = "disclosure-body";
		while (summary.nextSibling) body.append(summary.nextSibling);
		details.append(body);
		let expanded = details.open, animation = null;
		const finish = () => {
			details.open = expanded;
			body.style.height = "";
			body.style.overflow = "";
			body.inert = false;
			animation = null;
		};
		summary.addEventListener("click", (event) => {
			event.preventDefault();
			const from = details.open ? body.getBoundingClientRect().height : 0;
			animation?.cancel();
			expanded = !expanded;
			if (reduced.matches) { finish(); return; }
			details.open = true;
			body.inert = !expanded;
			body.style.overflow = "hidden";
			const to = expanded ? body.scrollHeight : 0;
			animation = body.animate([{ height: `${from}px`, opacity: from ? 1 : 0 }, { height: `${to}px`, opacity: expanded ? 1 : 0 }], { duration: 220, easing: "cubic-bezier(.2,.7,.2,1)", fill: "forwards" });
			animation.onfinish = () => { animation.cancel(); finish(); };
		});
		reduced.addEventListener("change", () => { animation?.cancel(); finish(); });
	});
}

function initializeGuide() {
	const copy = {
		resultados: ["03", "Resultados práticos", "O que os 558 casos mostram na prática."],
		metodo: ["04", "O algoritmo", "As ideias geométricas por trás da busca."],
		pesquisa: ["05", "O problema e a pesquisa", "O que está sendo resolvido e por quê."],
	};
	const ids = ["desafio", "historia", "resultados", "metodo", "pesquisa", "contato"];
	const nodes = new Map();
	for (const id of ids) {
		let node = element(id);
		if (id === "pesquisa" && !node) node = document.querySelector(".research-section");
		if (!node) continue;
		if (!node.id) node.id = id;
		node.dataset.guideSection = id;
		if (node.matches("section")) {
			node.classList.add("guided-section");
			const [number, title, subtitle] = copy[id] || ["", id, ""];
			const toggle = document.createElement("button");
			toggle.type = "button";
			toggle.className = "guided-summary guided-toggle";
			toggle.dataset.guideToggle = id;
			toggle.innerHTML = `<span><b>${number}</b><strong>${title}</strong><small>${subtitle}</small></span><em data-guide-status="${id}">Ainda não visitada</em>`;
			const panel = document.createElement("div");
			panel.className = "guided-panel";
			while (node.firstChild) panel.append(node.firstChild);
			node.append(toggle, panel);
			panel.hidden = true;
			toggle.setAttribute("aria-expanded", "false");
		}
		nodes.set(id, node);
	}
	if (!nodes.size) return;
	const storageKey = "tpp-siicusp34-visited-sections";
	let visited = new Set();
	try {
		visited = new Set(JSON.parse(localStorage.getItem(storageKey) || "[]").filter((id) => nodes.has(id)));
	} catch {
		visited = new Set();
	}
	const statusText = (id) => visited.has(id) ? "Visitada" : "Ainda não visitada";
	function update(id) {
		const node = nodes.get(id);
		if (!node) return;
		const isOpen = node.matches("details") ? node.open : !node.querySelector(":scope > .guided-panel")?.hidden;
		node.classList.toggle("is-open", isOpen);
		node.classList.toggle("is-visited", visited.has(id));
		document.querySelectorAll(`[data-guide-status="${id}"]`).forEach((status) => { status.textContent = statusText(id); });
		document.querySelectorAll(`[data-guide-target="${id}"]`).forEach((link) => link.classList.toggle("is-visited", visited.has(id)));
		const toggle = node.querySelector(":scope > .guided-toggle");
		if (toggle) toggle.setAttribute("aria-expanded", String(isOpen));
	}
	function markVisited(id) {
		if (!nodes.has(id)) return;
		visited.add(id);
		try { localStorage.setItem(storageKey, JSON.stringify([...visited])); } catch { /* Private browsing can reject storage. */ }
		update(id);
		const count = element("guide-title")?.closest(".guide-nav")?.querySelector("[data-guide-count]");
		if (count) count.textContent = `${visited.size}/${nodes.size} seções visitadas`;
	}
	function open(id, scroll = false) {
		const node = nodes.get(id);
		if (!node) return;
		if (node.matches("details")) {
			if (!node.open) node.querySelector(":scope > .guided-summary")?.click();
		}
		else {
			const panel = node.querySelector(":scope > .guided-panel");
			if (panel) panel.hidden = false;
		}
		markVisited(id);
		if (scroll) requestAnimationFrame(() => node.scrollIntoView({ behavior: "smooth", block: "start" }));
	}
	for (const [id, node] of nodes) {
		if (node.matches("details")) {
			node.querySelector(":scope > .guided-summary")?.addEventListener("click", () => { if (!node.open) markVisited(id); });
			node.addEventListener("toggle", () => { if (node.open) markVisited(id); update(id); });
		} else {
			node.querySelector(":scope > .guided-toggle")?.addEventListener("click", () => {
				const panel = node.querySelector(":scope > .guided-panel");
				const shouldOpen = Boolean(panel?.hidden);
				if (panel) panel.hidden = !shouldOpen;
				if (shouldOpen) markVisited(id); else update(id);
			});
		}
		update(id);
	}
	document.querySelectorAll("[data-guide-target]").forEach((link) => link.addEventListener("click", (event) => {
		const id = link.dataset.guideTarget;
		if (!nodes.has(id)) return;
		event.preventDefault();
		open(id, true);
		window.history.replaceState(null, "", `#${id}`);
	}));
	const count = element("guide-title")?.closest(".guide-nav")?.querySelector("[data-guide-count]");
	if (count) count.textContent = `${visited.size}/${nodes.size} seções visitadas`;
	const initial = window.location.hash.slice(1);
	if (nodes.has(initial)) open(initial, true);
}

function initializeContents() {
	const handle = element("toc-handle"), dialog = element("mobile-toc");
	const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
	const mobile = window.matchMedia("(max-width: 850px)");
	let animation = null;
	function open() {
		if (dialog.open) return;
		animation?.cancel();
		dialog.showModal();
		handle.setAttribute("aria-expanded", "true");
		if (!reduced.matches) animation = dialog.animate([{ transform: "translateX(100%)" }, { transform: "translateX(0)" }], { duration: 200, easing: "ease-out" });
		element("toc-close").focus();
	}
	function close(after) {
		animation?.cancel();
		const finish = () => { dialog.close(); handle.setAttribute("aria-expanded", "false"); after?.(); };
		if (reduced.matches) { finish(); return; }
		animation = dialog.animate([{ transform: "translateX(0)" }, { transform: "translateX(100%)" }], { duration: 180, easing: "ease-in" });
		animation.onfinish = finish;
	}
	handle.hidden = false;
	handle.addEventListener("click", open);
	element("toc-close").addEventListener("click", () => close());
	dialog.addEventListener("cancel", (event) => { event.preventDefault(); close(); });
	dialog.addEventListener("close", () => handle.setAttribute("aria-expanded", "false"));
	dialog.addEventListener("click", (event) => {
		const link = event.target.closest('a[href^="#"]');
		if (link) {
			event.preventDefault();
			const target = document.getElementById(link.hash.slice(1));
			close(() => {
				if (target.matches("details") && !target.open) target.querySelector("summary").click();
				target.scrollIntoView({ behavior: reduced.matches ? "instant" : "smooth" });
				const focusTarget = target.matches("details") ? target.querySelector("summary") : target;
				if (!focusTarget.hasAttribute("tabindex")) focusTarget.setAttribute("tabindex", "-1");
				focusTarget.focus({ preventScroll: true });
				dialog.querySelectorAll("a").forEach(item => item.removeAttribute("aria-current"));
				link.setAttribute("aria-current", "location");
			});
		} else if (event.target === dialog) {
			const box = dialog.getBoundingClientRect();
			if (event.clientX < box.left) close();
		}
	});
	for (const surface of [handle, dialog]) {
		let start = null;
		surface.addEventListener("pointerdown", event => { if (event.isPrimary) { start = [event.clientX, event.clientY]; if (surface === handle) surface.setPointerCapture(event.pointerId); } });
		surface.addEventListener("pointerup", event => {
			if (!start) return;
			const dx = event.clientX - start[0], dy = event.clientY - start[1];
			start = null;
			if (Math.abs(dx) < 35 || Math.abs(dx) < Math.abs(dy) * 1.5) return;
			if (surface === handle && dx < 0) open();
			if (surface === dialog && dx > 0) close();
		});
		surface.addEventListener("pointercancel", () => { start = null; });
	}
	mobile.addEventListener("change", () => { if (!mobile.matches && dialog.open) { animation?.cancel(); dialog.close(); } });
}

try { initialize(); }
catch (error) {
	element("event-error").hidden = false;
	element("event-error").textContent = "Não foi possível iniciar os controles interativos. Os resultados e a visualização estática continuam disponíveis. Recarregue a página para tentar novamente.";
	console.error(error);
}
