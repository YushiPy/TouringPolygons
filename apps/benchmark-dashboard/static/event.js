import { escapeHTML } from "./dom.js";
import { convexHull, endpointOffset, filterRows, gapRatio, pathPrefix, projectedCase, qualityLabel, regionColors, sortRows } from "./event-geometry.js";

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
	const speeds = [.25, .5, 1, 1.5, 2, 3, 4];
	let speedIndex = 2;
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
		element("show-labels").setAttribute("aria-pressed", String(row.polygons <= 15));
		element("case-select").value = row.case;
		element("case-picker-value").textContent = `Caso ${String(row.case).padStart(2, "0")} · ${row.polygons} regiões · ${row.exact ? "gap fechado" : "limite de tempo"}`;
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
		const rows = sortRows(filterRows(data.rows, element("result-filter").value, element("case-search").value), sorting.result.key, sorting.result.descending);
		element("result-count").textContent = `${rows.length} de ${data.rows.length} resultados. Selecione um caso para ver o caminho.`;
		element("result-rows").innerHTML = rows.length ? rows.map((item) => `<tr><th scope="row">${String(item.case).padStart(2, "0")}</th><td>${item.polygons}</td><td>${item.seconds < .001 ? "< 0,001" : number(item.seconds, 3)}</td><td>${escapeHTML(percent(gapRatio(item)))}</td><td><span class="status ${item.exact ? "certified" : "limited"}">${item.exact ? "✓ Gap fechado" : "◷ Limite de tempo"}</span></td><td><button type="button" data-open-case="${item.case}" aria-label="Ver caminho do caso ${item.case}">Ver caminho ↗</button></td></tr>`).join("") : '<tr><td colspan="6">Nenhum caso corresponde à busca. Limpe o número ou escolha “Todos os resultados”.</td></tr>';
	}

	document.querySelectorAll("button:disabled, input:disabled, select:disabled").forEach((control) => { control.disabled = false; });
	document.querySelectorAll(".example").forEach((button) => button.addEventListener("click", () => selectCase(Number(button.dataset.case))));
	element("case-select").addEventListener("change", (event) => selectCase(Number(event.target.value)));
	element("show-labels").addEventListener("click", () => { element("show-labels").setAttribute("aria-pressed", String(!enabled("show-labels"))); draw(); });
	element("show-hulls").addEventListener("click", () => { element("show-hulls").setAttribute("aria-pressed", String(!enabled("show-hulls"))); draw(); });
	element("show-contacts").addEventListener("click", () => { element("show-contacts").setAttribute("aria-pressed", String(!enabled("show-contacts"))); draw(); });
	element("show-decomposition").addEventListener("click", () => { element("show-decomposition").setAttribute("aria-pressed", String(!enabled("show-decomposition"))); draw(); });
	element("zoom-in").addEventListener("click", () => { zoom = Math.min(3, zoom + .5); draw(); });
	element("zoom-out").addEventListener("click", () => { zoom = Math.max(1, zoom - .5); draw(); });
	element("fit-view").addEventListener("click", () => { zoom = 1; pan = [0, 0]; draw(); });
	const pointers = new Map();
	let gesture = null;
	function localPoint(event) {
		const box = map.getBoundingClientRect();
		const scale = Math.min(box.width / 840, box.height / 480);
		return [(event.clientX - box.left - box.width / 2) / scale, (event.clientY - box.top - box.height / 2) / scale];
	}
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
			fraction = Math.min(1, fraction + (now - previous) * speeds[speedIndex] / 8000);
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
		element("case-picker-button").focus({ preventScroll: true });
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
	for (const prefix of ["picker", "result"]) {
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
}

try { initialize(); }
catch (error) {
	element("event-error").hidden = false;
	element("event-error").textContent = "Não foi possível iniciar os controles interativos. Os resultados e a visualização estática continuam disponíveis. Recarregue a página para tentar novamente.";
	console.error(error);
}
