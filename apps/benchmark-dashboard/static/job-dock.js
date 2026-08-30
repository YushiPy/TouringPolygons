import { $, escapeHTML, setOutput } from "./dom.js";
import { formatElapsed } from "./format.js";
import { jobDockStatusClass, jobKindLabel, jobPanel, jobProgressLabel, jobTerminalState } from "./job-utils.js";
import { closeIconSVG, setCloseIcon, warningIconSVG } from "./ui-utils.js";

function runningJobs(state) {
	return state.recentJobs.filter((job) => job.status === "running" || job.status === "stopping");
}

export function dismissFinishedJobForPanel(state, panelId, renderJobDock) {
	if (state.finishedDockJob && jobPanel(state.finishedDockJob) === panelId) {
		const dock = $("#job-dock");
		dock?.classList.add("is-dismissing");
		setTimeout(() => {
			state.finishedDockJob = null;
			renderJobDock();
		}, 180);
	}
}

function jobDockStatusIcon(job) {
	const state = jobTerminalState(job);
	if (state === "failed" || state === "canceled") {
		return `<span class="job-dock-warning" aria-hidden="true">${warningIconSVG()}</span>`;
	}
	return '<span class="job-dock-check" aria-hidden="true">✓</span>';
}

export function createJobDock({ requestJSON, state, switchPanel }) {
	async function cancelJobId(jobId) {
		if (!jobId) {
			return;
		}
		try {
			await requestJSON(`/api/jobs/${jobId}/cancel`, { method: "POST" });
		} catch (error) {
			setOutput($("#run-output"), error.message);
		}
	}

	function updateJobDockItem(item, job) {
		if (!item) {
			return;
		}
		const active = job.status === "running" || job.status === "stopping";
		item.dataset.jobPanel = jobPanel(job);
		item.classList.toggle("is-active", active);
		item.classList.toggle("is-complete", !active);
		item.classList.toggle("is-completed", jobTerminalState(job) === "completed");
		item.classList.toggle("is-failed", jobTerminalState(job) === "failed");
		item.classList.toggle("is-canceled", jobTerminalState(job) === "canceled");
		item.querySelector("[data-job-dock-status]").textContent = `${jobKindLabel(job)} ${active ? "running..." : job.status}`;
		item.querySelector("[data-job-dock-campaign]").textContent = job.campaign || "-";
		item.querySelector("[data-job-dock-progress]").textContent = `${jobProgressLabel(job)} | ${formatElapsed((job.finished_at || Date.now() / 1000) - (job.started_at || Date.now() / 1000))}`;
	}

	function renderJobDock(previousJobs = []) {
		const dock = $("#job-dock");
		if (!dock) {
			return;
		}
		const previousActive = new Map(previousJobs
			.filter((job) => job.status === "running" || job.status === "stopping")
			.map((job) => [job.id, job]));
		for (const job of state.recentJobs) {
			if (previousActive.has(job.id) && job.status !== "running" && job.status !== "stopping") {
				state.finishedDockJob = job;
			}
		}
		const activeJobs = runningJobs(state);
		const visibleJobs = activeJobs.length > 0 ? activeJobs : state.finishedDockJob ? [state.finishedDockJob] : [];
		dock.classList.toggle("is-hidden", visibleJobs.length === 0);
		dock.classList.remove("is-dismissing");
		const signature = visibleJobs.map((job) => `${job.id}:${job.status}`).join(",");
		if (dock.dataset.signature === signature) {
			visibleJobs.forEach((job) => updateJobDockItem(dock.querySelector(`[data-job-id="${CSS.escape(job.id)}"]`), job));
			return;
		}
		dock.dataset.signature = signature;
		dock.innerHTML = visibleJobs.map((job) => {
			const active = job.status === "running" || job.status === "stopping";
			return `
			<div class="job-dock-item ${active ? "is-active" : `is-complete ${jobDockStatusClass(job)}`}" data-job-panel="${jobPanel(job)}" data-job-id="${escapeHTML(job.id)}">
				<button class="job-dock-main" type="button" data-job-open>
					<span data-job-dock-status>${jobKindLabel(job)} ${active ? "running..." : job.status}</span>
					<strong data-job-dock-campaign>${escapeHTML(job.campaign || "-")}</strong>
					<small data-job-dock-progress>${escapeHTML(jobProgressLabel(job))} | ${formatElapsed((job.finished_at || Date.now() / 1000) - (job.started_at || Date.now() / 1000))}</small>
				</button>
				${active ? `<button class="job-dock-stop" type="button" data-job-stop aria-label="Stop job">${closeIconSVG()}</button>` : jobDockStatusIcon(job)}
			</div>
		`;
		}).join("");
		dock.querySelectorAll("[data-job-open]").forEach((button) => {
			button.addEventListener("click", (event) => {
				event.stopPropagation();
				openJobDockItem(button.closest(".job-dock-item"));
			});
		});
		dock.querySelectorAll(".job-dock-item").forEach((item) => {
			item.addEventListener("click", (event) => {
				if (event.target.closest("[data-job-stop]")) {
					return;
				}
				openJobDockItem(item);
			});
		});
		dock.querySelectorAll("[data-job-stop]").forEach((button) => {
			button.addEventListener("click", async (event) => {
				event.stopPropagation();
				const item = button.closest(".job-dock-item");
				button.disabled = true;
				setCloseIcon(button);
				await cancelJobId(item.dataset.jobId);
			});
		});
	}

	function openJobDockItem(item) {
		if (!item || $("#job-dock")?.dataset.suppressClick === "true") {
			return;
		}
		switchPanel(item.dataset.jobPanel);
		if (item.classList.contains("is-complete")) {
			item.classList.add("is-dismissing");
			setTimeout(() => {
				state.finishedDockJob = null;
				renderJobDock();
			}, 180);
		}
	}

	function setupJobDockDrag() {
		const dock = $("#job-dock");
		if (!dock) {
			return;
		}
		let drag = null;
		dock.addEventListener("pointerdown", (event) => {
			if (event.target.closest("button:not(.job-dock-main)")) {
				return;
			}
			event.preventDefault();
			const rect = dock.getBoundingClientRect();
			drag = {
				pointerId: event.pointerId,
				dx: event.clientX - rect.left,
				dy: event.clientY - rect.top,
				startX: event.clientX,
				startY: event.clientY,
				moved: false,
				target: event.target.closest(".job-dock-item"),
			};
			dock.setPointerCapture(event.pointerId);
		});
		dock.addEventListener("pointermove", (event) => {
			if (!drag || drag.pointerId !== event.pointerId) {
				return;
			}
			if (Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) > 10) {
				drag.moved = true;
				dock.dataset.dragging = "true";
			}
			if (!drag.moved) {
				return;
			}
			const left = Math.max(8, Math.min(window.innerWidth - dock.offsetWidth - 8, event.clientX - drag.dx));
			const top = Math.max(8, Math.min(window.innerHeight - dock.offsetHeight - 8, event.clientY - drag.dy));
			dock.style.left = `${left}px`;
			dock.style.top = `${top}px`;
			dock.style.right = "auto";
			dock.style.bottom = "auto";
		});
		const finish = (event) => {
			if (!drag || drag.pointerId !== event.pointerId) {
				return;
			}
			if (drag.moved) {
				dock.dataset.suppressClick = "true";
				setTimeout(() => {
					delete dock.dataset.suppressClick;
				}, 120);
			} else if (!event.target.closest("[data-job-stop]")) {
				openJobDockItem(drag.target);
			}
			delete dock.dataset.dragging;
			drag = null;
			dock.releasePointerCapture?.(event.pointerId);
		};
		dock.addEventListener("pointerup", finish);
		dock.addEventListener("pointercancel", finish);
	}

	return {
		cancelJobId,
		renderJobDock,
		setupJobDockDrag,
	};
}
