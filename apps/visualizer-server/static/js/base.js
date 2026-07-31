
function enableSmartNavigation(form) {

	const inputs = [...form.querySelectorAll("input")];
	const submitButton = form.querySelector("button");

	function shiftFocus(newIndex) {
		if (newIndex >= 0 && newIndex < inputs.length) {
			inputs[newIndex].focus();
		} else if (newIndex === inputs.length) {
			submitButton.focus();
		}
	}

	function getShift(event) {

		if (event.key === "Enter") {
			return event.shiftKey ? -1 : 1;
		} else if (event.key === "ArrowUp") {
			return -1;
		} else if (event.key === "ArrowDown") {
			return 1;
		} else {
			return 0;
		}
	}

	function handleKeyDown(event) {

		const shift = getShift(event);
		const newIndex = inputs.indexOf(event.target) + shift;

		if (shift !== 0) {

			event.preventDefault();
			shiftFocus(newIndex);

			if (event.key === "Enter" && !event.shiftKey && newIndex === inputs.length) {
				submitButton.click();
			}
		}
	}

	inputs.forEach(input => {
		if (!input.hasAttribute("data-smart-nav")) {
			input.addEventListener("keydown", handleKeyDown);
			input.setAttribute("data-smart-nav", "true");
		}
	})
}

const smartFormClass = ".smart-form";

function initSmartForms() {
	document.querySelectorAll(smartFormClass).forEach(enableSmartNavigation);
}

document.addEventListener("DOMContentLoaded", initSmartForms);
document.addEventListener("htmx:afterSwap", initSmartForms);
