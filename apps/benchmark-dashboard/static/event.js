import { escapeHTML } from "./dom.js";
import { convexHull, endpointOffset, filterRows, gapRatio, pathPrefix, projectedCase, qualityLabel, regionColors, sortRows, sortGroupedRows, toggleOrderRegion, playbackDuration } from "./event-geometry.js?v=20260907-3";

const element = (id) => document.getElementById(id);
const number = (value, digits = 4) => value.toLocaleString("pt-BR", { maximumFractionDigits: digits });
const percent = (value) => value < 0.000001 ? "< 0,0001%" : `${number(value * 100, value < .0001 ? 4 : 2)}%`;
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
	const speeds = [.25, .5, 1, 1.5, 2, 3, 4];
	let speedIndex = 2;
	let resultGroup = null;
	let showAllResults = false;
	const sorting = { picker: { key: "case", descending: false }, result: { key: "case", descending: false } };
	const enabled = (id) => element(id).getAttribute("aria-pressed") === "true";
	const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
	const mapContent = element("map-content");
	const map = element("route-map");

	function camera() {
		pan = [Math.max(-420 * (zoom - 1), Math.min(420 * (zoom - 1), pan[0])), Math.max(-240 * (zoom - 1), Math.min(240 * (zoom - 1), pan[1]))];
		mapContent.setAttribute("transform", `translate(${420 + pan[0]} ${240 + pan[1]}) scale(${zoom}) translate(-420 -240)`);
		map.classList.toggle("is-zoomed", zoom > 1);
		map.style.touchAction = "none";
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
			${projected.polygons.map((polygon, index) => `<polygon class="region" data-region="${index}" points="${coordinates(polygon)}"><title>Região ${index + 1}; ${row.order.indexOf(index) + 1}ª na ordem exportada</title></polygon>`).join("")}
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
		element("map-caption").textContent = decomposition ? "Decomposição convexa calculada pela implementação C++ autoral de Greene usada pelo solver. As linhas tracejadas delimitam as peças, não os ramos percorridos na busca." : hulls ? "Os contornos tracejados são fechos convexos: regiões maiores que simplificam o problema para obter limites inferiores." : "Basta tocar a borda ou atravessar a região. Seus centros não são pontos de visita obrigatórios.";
		if (enabled("show-contacts")) element("map-caption").textContent += " Os pontos marcam os primeiros contatos reconstruídos do caminho, com tolerância de 10⁻⁷. Visitas simultâneas podem compartilhar um ponto.";
		element("map-title").textContent = `Caso ${row.case}: caminho verificado por ${row.polygons} regiões; ${row.exact ? "gap numérico fechado" : "limite de tempo"}.`;
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
		element("show-labels").setAttribute("aria-pressed", String(row.polygons <= 15));
		element("case-select").value = row.case;
		element("case-picker-value").textContent = `Caso ${String(row.case).padStart(2, "0")} · ${row.polygons} regiões · ${row.exact ? "gap fechado" : "limite de tempo"}`;
		document.querySelectorAll(".example").forEach((button) => {
			const active = Number(button.dataset.case) === row.case;
			button.classList.toggle("active", active);
			button.setAttribute("aria-pressed", String(active));
		});
		element("drawing-title").textContent = titles[row.case] || `${row.polygons} regiões, extremos fixos`;
		element("speed-value").title = `A 1×, este caso leva ${number(playbackDuration(row.polygons) / 1000, 1)} s para percorrer o caminho.`;
		element("case-id").textContent = `Caso ${String(row.case).padStart(2, "0")}`;
		element("outcome-badge").className = `status ${row.exact ? "certified" : "limited"}`;
		element("outcome-badge").textContent = row.exact ? "✓ Gap numérico fechado" : "◷ Encerrado por tempo";
		element("outcome-title").textContent = row.exact ? "Ótimo certificado" : "Caminho encontrado; ótimo ainda não certificado";
		element("outcome-explanation").textContent = row.exact ? "Dentro das tolerâncias numéricas adotadas. O comprimento encontrado e seu limite inferior concordam nessas tolerâncias." : "O caminho visita todas as regiões. A diferença entre os limites ainda não permite certificar o ótimo.";
		element("case-length").textContent = number(row.upper_bound, 2);
		element("case-time").textContent = row.seconds < .001 ? "< 0,001 s" : `${number(row.seconds, 3)} s`;
		element("case-regions").textContent = `${row.polygons} / ${row.polygons}`;
		element("case-gap").textContent = percent(gapRatio(row));
		element("case-lower").textContent = row.lower_bound.toPrecision(17);
		element("case-upper").textContent = row.upper_bound.toPrecision(17);
		element("case-order").textContent = row.order.map((index, position) => `${position + 1}ª → região ${index + 1}`).join(" · ");
		element("case-quality").textContent = qualityLabel(row);
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
		const rows = sortGroupedRows(filterRows(data.rows, "all", element("case-search").value), sorting.result.key, sorting.result.descending, resultGroup);
		const visible = showAllResults ? rows : rows.slice(0, 8);
		element("result-count").textContent = `${visible.length} de ${rows.length} resultados${rows.length < data.rows.length ? " nesta busca" : ""}. Selecione um caso para ver o caminho.`;
		element("show-all-results").hidden = rows.length <= 8;
		element("show-all-results").textContent = showAllResults ? "Mostrar menos resultados" : `Ver todos os ${rows.length} resultados`;
		element("show-all-results").setAttribute("aria-expanded", String(showAllResults));
		element("result-rows").innerHTML = visible.length ? visible.map((item) => `<tr><th scope="row"><span class="mobile-case-label">Caso </span>${String(item.case).padStart(2, "0")}</th><td data-label="Regiões">${item.polygons}</td><td data-label="Tempo (s)">${item.seconds < .001 ? "< 0,001" : number(item.seconds, 3)}</td><td data-label="Gap relativo">${escapeHTML(percent(gapRatio(item)))}</td><td class="result-status"><span class="status ${item.exact ? "certified" : "limited"}">${item.exact ? "✓ Ótimo certificado" : "◷ Limite de tempo"}</span></td><td class="result-action"><button type="button" data-open-case="${item.case}" aria-label="Ver caminho do caso ${item.case}">Ver caminho →</button></td></tr>`).join("") : '<tr><td colspan="6">Nenhum caso corresponde à busca. Limpe o número para ver todos os casos.</td></tr>';
	}

	document.querySelectorAll("button:disabled, input:disabled, select:disabled").forEach((control) => { control.disabled = false; });
	document.querySelectorAll(".example").forEach((button) => button.addEventListener("click", () => selectCase(Number(button.dataset.case))));
	element("case-select").addEventListener("change", (event) => selectCase(Number(event.target.value)));
	element("show-labels").addEventListener("click", () => { element("show-labels").setAttribute("aria-pressed", String(!enabled("show-labels"))); draw(); });
	element("show-hulls").addEventListener("click", () => { element("show-hulls").setAttribute("aria-pressed", String(!enabled("show-hulls"))); draw(); });
	element("show-contacts").addEventListener("click", () => { element("show-contacts").setAttribute("aria-pressed", String(!enabled("show-contacts"))); draw(); });
	element("show-decomposition").addEventListener("click", () => { element("show-decomposition").setAttribute("aria-pressed", String(!enabled("show-decomposition"))); draw(); });
	element("fit-view").addEventListener("click", () => { zoom = 1; pan = [0, 0]; draw(); });
	const pointers = new Map();
	let gesture = null;
	function localPoint(event) {
		const box = map.getBoundingClientRect();
		const scale = Math.min(box.width / 840, box.height / 480);
		return [(event.clientX - box.left - box.width / 2) / scale, (event.clientY - box.top - box.height / 2) / scale];
	}
	function zoomAt(next, point) {
		const previous = zoom;
		zoom = Math.max(1, Math.min(3, next));
		pan = point.map((value, axis) => value - (value - pan[axis]) * zoom / previous);
		draw();
	}
	let trackpadGesture = null;
	map.addEventListener("wheel", (event) => {
		event.preventDefault();
		if (trackpadGesture) return;
		const units = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 480 : 1;
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
			zoom = Math.max(1, Math.min(3, gesture.zoom * distance / Math.max(gesture.distance, 1e-6)));
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
		else if (event.key === "+" || event.key === "=") zoom = Math.min(3, zoom + .5);
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
	document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); });
	reducedMotion.addEventListener("change", stop);
	element("case-search").addEventListener("input", () => { showAllResults = false; renderTable(); });
	element("show-all-results").addEventListener("click", () => { showAllResults = !showAllResults; renderTable(); });
	element("result-rows").addEventListener("click", (event) => {
		const button = event.target.closest("[data-open-case]");
		if (!button) return;
		selectCase(Number(button.dataset.openCase));
		element("explorar").scrollIntoView({ behavior: reducedMotion.matches ? "instant" : "smooth" });
		element("play-route").focus({ preventScroll: true });
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
		const rows = sortRows(filterRows(data.rows, "all", element("picker-search").value), sorting.picker.key, sorting.picker.descending);
		element("picker-count").textContent = `${rows.length} casos disponíveis`;
		element("picker-options").innerHTML = rows.length ? rows.map((item) => `<button type="button" data-pick-case="${item.case}" aria-pressed="${item.case === row.case}"><span><strong>Caso ${String(item.case).padStart(2, "0")}</strong><small>${item.polygons} regiões · ${number(item.seconds, 3)} s · gap ${percent(gapRatio(item))}</small></span><span class="status ${item.exact ? "certified" : "limited"}">${item.exact ? "✓ Gap fechado" : "◷ Limite de tempo"}</span></button>`).join("") : "<p>Nenhum caso encontrado. Experimente outro número.</p>";
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
		if (button) { selectCase(Number(button.dataset.pickCase)); picker.close(); }
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
		const names = { case: "caso", polygons: "regiões", seconds: "tempo", gap: "gap relativo" };
		document.querySelectorAll("[data-result-column]").forEach((header) => {
			const key = header.dataset.resultColumn;
			const active = key === "result" ? resultGroup !== null : key === sorting.result.key;
			const descending = key === "result" ? resultGroup : sorting.result.descending;
			const primary = resultGroup === null ? key === sorting.result.key : key === "result";
			header.setAttribute("aria-sort", active && primary ? (descending ? "descending" : "ascending") : "none");
			header.querySelector(".sort-arrow").textContent = active ? (descending ? " ↓" : " ↑") : "";
			header.classList.toggle("sort-active", active);
		});
		element("result-sort-description").textContent = `${resultGroup === null ? "Sem agrupamento; " : resultGroup ? "Limites de tempo primeiro; depois " : "Ótimos certificados primeiro; depois "}${names[sorting.result.key]} em ordem ${sorting.result.descending ? "decrescente" : "crescente"}. Resultado alterna: ótimos primeiro, tempo primeiro, sem agrupamento.`;
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
	element("download-case").addEventListener("click", () => download(`tpp-siicusp-caso-${row.case}.json`, JSON.stringify({ provenance: data.provenance, config: data.config, result: row }, null, 2), "application/json"));
	element("download-results").addEventListener("click", () => {
		const header = ["case", "polygons", "seconds", "lower_bound", "upper_bound", "relative_gap", "exact", "termination", "valid", "sha256"];
		const csv = [header.join(","), ...data.rows.map((item) => header.map((key) => key === "relative_gap" ? gapRatio(item) : item[key]).join(","))].join("\n");
		download("siicusp34-resultados-60-casos.csv", csv, "text/csv;charset=utf-8");
	});
	const requested = new URLSearchParams(window.location.search).get("caso");
	selectCase(requested !== null && /^\d+$/.test(requested) ? Number(requested) : 2, false) || selectCase(2, false);
	renderTable();
	initializeChallenge();
	initializePieceChallenge();
	initializeDisclosures();
	initializeContents();
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
		}).join("") + (compared ? `<polyline class="challenge-reference" points="${coordinates(best.path)}"/><polyline class="route-line" points="${coordinates(chosen.path)}"/>` : orderSketch("order", challenge.geometry, order.map((index) => challenge.geometry.polygons[index]))) + '<circle cx="25" cy="130" r="5" fill="white"/><text x="17" y="153">S</text><circle cx="395" cy="130" r="5" fill="#ffad66"/><text x="389" y="153">T</text>';
		element("challenge-choices").innerHTML = letters.map((letter, index) => `<button type="button" data-challenge-region="${index}" aria-pressed="${order.includes(index)}">${letter}</button>`).join("");
		element("challenge-order").textContent = `Sua ordem: S → ${order.length ? order.map((index) => letters[index]).join(" → ") + " → " : ""}${order.length < 4 ? "… → " : ""}T`;
		element("challenge-compare").disabled = order.length !== 4 || compared;
		element("challenge-undo").disabled = !order.length;
		element("challenge-reset").disabled = !order.length;
		element("challenge-feedback").innerHTML = compared ? `<strong>${chosen.length - best.length < .00001 ? "Você encontrou uma das melhores ordens!" : `Seu caminho ficou ${number(100 * (chosen.length / best.length - 1), 1)}% mais longo.`}</strong><div class="challenge-scores"><span>Sua escolha <b>${number(chosen.length, 2)}</b></span><span>Melhor das 24 <b>${number(best.length, 2)}</b></span></div><p>Comprimentos em unidades deste exemplo. Laranja: sua escolha; tracejado claro: referência (${best.order.map((index) => letters[index]).join(" → ")}).</p>` : "";
	}
	function choose(event) {
		const control = event.target.closest("[data-challenge-region]");
		if (!control) return;
		const index = Number(control.dataset.challengeRegion);
		order = toggleOrderRegion(order, index);
		compared = false;
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
	element("challenge-compare").addEventListener("click", () => { compared = true; render(); });
	element("challenge-undo").addEventListener("click", () => { order.pop(); compared = false; render(); });
	element("challenge-reset").addEventListener("click", () => { order = []; compared = false; render(); element("challenge-choices").querySelector("button").focus({ preventScroll: true }); });
	render();
}

function initializePieceChallenge() {
	const data = JSON.parse(element("challenge-data").textContent).piece_challenge;
	const best = data.solutions[data.reference];
	const letters = ["A", "B", "C"];
	let choices = [null, null, null], compared = false;
	const map = element("piece-map");
	function render() {
		const selected = data.solutions.find((item) => item.choices.every((piece, index) => piece === choices[index]));
		map.innerHTML = data.pieces.map((pieces, region) => pieces.map((piece, index) => {
			const x = (piece[0][0] + piece[2][0]) / 2, y = (piece[0][1] + piece[2][1]) / 2;
			return `<g role="button" tabindex="0" data-piece="${index}" data-piece-region="${region}" aria-label="Região ${letters[region]}, peça ${index + 1}" aria-pressed="${choices[region] === index}"><polygon class="challenge-region ${choices[region] === index ? "chosen" : ""}" points="${coordinates(piece)}"/><text x="${x}" y="${y}" dominant-baseline="central" text-anchor="middle">${index + 1}</text></g>`;
		}).join("") + `<text x="${data.pieces[region][0][0][0] + 40}" y="${data.pieces[region][0][0][1] - 10}" text-anchor="middle">${letters[region]}</text>`).join("") + (compared ? `<polyline class="challenge-reference" points="${coordinates(best.path)}"/><polyline class="route-line" points="${coordinates(selected.path)}"/>` : orderSketch("pieces", data.geometry, choices.slice(0, choices.includes(null) ? choices.indexOf(null) : choices.length).map((piece, region) => data.pieces[region][piece]))) + '<circle cx="25" cy="200" r="5" fill="white"/><text x="17" y="223">S</text><circle cx="395" cy="180" r="5" fill="#ffad66"/><text x="389" y="203">T</text>';
		element("piece-choices").innerHTML = letters.map((letter, region) => `<fieldset><legend>Região ${letter}</legend>${[0, 1, 2].map((piece) => `<button type="button" data-piece="${piece}" data-piece-region="${region}" aria-pressed="${choices[region] === piece}" aria-label="${letter}: peça ${piece + 1}">${piece + 1}</button>`).join("")}</fieldset>`).join("");
		element("piece-selection").textContent = choices.map((piece, region) => `${letters[region]}: ${piece === null ? "?" : `peça ${piece + 1}`}`).join(" · ");
		element("piece-compare").disabled = choices.includes(null) || compared;
		element("piece-reset").disabled = choices.every((piece) => piece === null);
		element("piece-feedback").innerHTML = compared ? `<strong>${selected.length - best.length < .00001 ? "Você encontrou uma das melhores combinações!" : `Seu caminho ficou ${number(100 * (selected.length / best.length - 1), 1)}% mais longo.`}</strong><div class="challenge-scores"><span>Sua escolha <b>${number(selected.length, 2)}</b></span><span>Melhor das 27 <b>${number(best.length, 2)}</b></span></div><p>Laranja: sua escolha. Tracejado claro: referência (${best.choices.map((piece, region) => `${letters[region]}${piece + 1}`).join(" → ")}). Comprimentos em unidades deste exemplo.</p>` : "";
	}
	function choose(event) {
		const control = event.target.closest("[data-piece]");
		if (!control) return;
		const region = Number(control.dataset.pieceRegion), piece = Number(control.dataset.piece);
		const parent = map.contains(control) ? map : element("piece-choices");
		choices[region] = piece;
		compared = false;
		render();
		if (event.type === "keydown" || event.detail === 0) parent.querySelector(`[data-piece-region="${region}"][data-piece="${piece}"]`).focus({ preventScroll: true });
	}
	map.addEventListener("click", choose);
	map.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(event); } });
	element("piece-choices").addEventListener("click", choose);
	element("piece-compare").addEventListener("click", () => { compared = true; render(); });
	element("piece-reset").addEventListener("click", () => { choices = [null, null, null]; compared = false; render(); element("piece-choices").querySelector("button").focus({ preventScroll: true }); });
	render();
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
