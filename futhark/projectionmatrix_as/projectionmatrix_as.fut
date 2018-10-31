import "futlib/array"
import "utils"
import "dda"
import "initialisation"

--just calls the algorithm no formatting!
let handleAngles (angle: f64) (detectorPositions: []f64) : [](f64, f64) =
	zip (replicate (length detectorPositions) angle) detectorPositions


let buildDDAParams (gridHalfWidth: f64) (delta: f64) (row: i32, angle: f64, detector: f64)
	: (f64, i16, i16, f64, bool, bool, i32) =
	let radAngle = toRad angle in
	let (cosTheta, sinTheta) = (f64.cos radAngle, f64.sin radAngle) in
	let (detX, detY) = (cosTheta*detector, sinTheta*detector) in
	let (curX, curY, chi, inBounds, discard) =
		getGridInterSection(gridHalfWidth, delta, detX, detY, angle) in
	(angle, i16.f64 curX, i16.f64 curY, chi, inBounds, discard, row)


-- inMode determines return format. 0 = COO, 1 = CSR
let main(angles: []f64, detectorPositions: []f64,
		 gridsize: i32) :
		 [](i32, (i32, [](i16, i16, f64))) =
	let delta = 1f64
	let rowOffset = 0i32
	let gridHalfWidth = (r64(gridsize)/2f64)
	-- Applies buildDDAParams partially with the arguments shared
	-- by all DDA instances.
	-- initFun is applied to each line, constructing the arguments needed
	-- by the DDA to run for that particular line.
	let initFun = buildDDAParams gridHalfWidth delta in
	--let n = f64.to_i32 ((gridHalfWidth/delta)*2.0*2.0) in
	let nFun = (\(angle) -> 2.0*gridHalfWidth/delta + f64.ceil(2.0*gridHalfWidth/delta*tan(toRad (toAxisAngle angle)))) in
	let n = f64.to_i32 (f64.maximum(map nFun angles)) in
	let ddaFun = DDA.run (n,
							i16.f64 (gridHalfWidth/delta),
							i16.f64 (gridHalfWidth/delta),
							delta) in

	-- Creates a cartesian product of angles and detectors in parallel
	let(eAngles, eDets) =
		(unzip (flatten (map (flip handleAngles detectorPositions) angles)))
	-- x is a list of tuples: rownumber * angle * detector
	-- for each angle for each detector
	let x =
		zip3 (map (+rowOffset) (iota (length(detectorPositions)*length(angles))))
				eAngles eDets
	in
	unsafe
		map initFun x |>
		filter (\(_,_,_,_,_,disc,_) -> !disc) |>
		map ddaFun
