import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import createModule from "../static/wasm/tpp_convex_wasm.js";

const wasmBinary = await readFile(new URL("../static/wasm/tpp_convex_wasm.wasm", import.meta.url));
const module = await createModule({ wasmBinary });
const box = (x, y, X, Y) => [[x, y], [X, y], [X, Y], [x, Y]];
const length = path => path.slice(1).reduce((sum, p, i) => sum + Math.hypot(p[0] - path[i][0], p[1] - path[i][1]), 0);

// Independent closed half-plane clipping and greedy nondecreasing visits.
function ordered(path, polygons) {
    let segment = 0, parameter = 0;
    for (const polygon of polygons) {
        const cross = (a, b) => a[0] * b[1] - a[1] * b[0];
        const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
        const area = polygon.reduce((sum, p, i) => sum + cross(p, polygon[(i + 1) % polygon.length]), 0);
        const orientation = Math.sign(area);
        let found = false;
        for (; segment < path.length - 1; ++segment, parameter = 0) {
            const a = path[segment], d = sub(path[segment + 1], a);
            let lo = parameter, hi = 1;
            for (let j = 0; j < polygon.length; ++j) {
                const p = polygon[j], edge = sub(polygon[(j + 1) % polygon.length], p);
                const constant = orientation * cross(edge, sub(a, p));
                const slope = orientation * cross(edge, d);
                const tolerance = 1e-12 * Math.hypot(...edge);
                if (slope > 0) lo = Math.max(lo, (-tolerance - constant) / slope);
                else if (slope < 0) hi = Math.min(hi, (-tolerance - constant) / slope);
                else if (constant < -tolerance) { lo = 1; hi = 0; break; }
            }
            if (lo <= hi) { parameter = lo; found = true; break; }
        }
        if (!found) return false;
    }
    return true;
}

function solve(s, t, polygons) {
    const flat = polygons.flat(2);
    const points = module._malloc(flat.length * 8);
    const sizes = module._malloc(polygons.length * 4);
    try {
        module.HEAPF64.set(flat, points / 8);
        module.HEAP32.set(polygons.map(p => p.length), sizes / 4);
        const count = module._tpp_solve(...s, ...t, points, sizes, polygons.length, 200000, 3);
        assert(count > 0, "WASM solver failed");
        assert.equal(module._tpp_solution_exact(), 1, "WASM result was incomplete");
        const pointer = module._tpp_get_path_points() / 8;
        return Array.from({ length: count }, (_, i) => [module.HEAPF64[pointer + 2 * i], module.HEAPF64[pointer + 2 * i + 1]]);
    } finally {
        module._free(points);
        module._free(sizes);
    }
}

const polygons = [box(0, -6, 6, 6), [[-8, -4], [8, 4], [8, 9], [-8, 9]]];
const cases = [
    ["simultaneous intersection", [-3, -4], [4, -3], polygons, 10],
    ["two reflections", [-3, -4], [-4, -3], polygons, 13 * Math.sqrt(10) / 5],
    ["ordered rectangles", [-4, -4], [4, -4], [box(0, 2, 2, 4), box(0, -2, 3, 2), box(-1, 3, 1, 4)], 2 * Math.sqrt(13) + 4 * Math.sqrt(5)],
    ["ray extension", [0, -2], [1, 3], [box(2, 3, 5, 4), box(-3, -2, 0, 2), box(2, 2, 3, 5)], Math.sqrt(29) + Math.sqrt(5) + Math.sqrt(10)],
    ["shared source", [0, 0], [3, -2], [box(-2, 0, 0, 2), box(0, 0, 2, 2)], Math.sqrt(13)],
];

const fixture = await readFile(new URL("../../../benchmarks/suites/intersection-audit/Wrong1.bin", import.meta.url));
let offset = 0;
const scalar = () => { const v = fixture.readDoubleLE(offset); offset += 8; return v; };
const count = () => { const v = Number(fixture.readBigUInt64LE(offset)); offset += 8; return v; };
const point = () => [scalar(), scalar()];
const s = point(), t = point();
const wrong1 = Array.from({ length: count() }, () => Array.from({ length: count() }, point));
const expected = Array.from({ length: count() }, point);
cases.push(["Wrong1 raw winding", s, t, wrong1, length(expected)]);

for (const [name, start, target, regions, optimum] of cases) {
    const path = solve(start, target, regions);
    assert.deepEqual(path[0], start, `${name}: source`);
    assert.deepEqual(path.at(-1), target, `${name}: target`);
    assert(ordered(path, regions), `${name}: ordered visitation`);
    assert(Math.abs(length(path) - optimum) <= 2e-12 * Math.max(1, optimum), `${name}: optimality`);
    console.log(`PASS ${name}: ${length(path)}`);
}
