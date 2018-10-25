import "futlib/array"
import "utils"
import "dda"
import "initialisation"


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

-- Translates the grid coordinates such that the grid has its lower left
-- corner at origo. Also flattens the grid coordinates, since the COO
-- expects them like that. returns parameters corresponding to a COO matrix
-- COO are nice because they are easy to convert to other sparse formats
let translateCOO (ghdelta: f64)
								 (row: i32, (_, ddaResult: [](i16, i16, f64)))
					: [](i32, i32, f64) =
	map (\(x, y, len) ->
				let ghdelta = f64.to_i32 ghdelta in
				let colOffset = ((i32.i16 x)+ghdelta) + ((i32.i16 y)+ghdelta)*(ghdelta*2)
				in
					(row, colOffset, len)
			) ddaResult

let translateCSR (ghdelta: f64)
								 (_, (_, ddaResult: [](i16, i16, f64)))
					: [](i32, f64) =
	map (\(x, y, len) ->
				let ghdelta = f64.to_i32 ghdelta in
				let colOffset = ((i32.i16 x)+ghdelta) + ((i32.i16 y)+ghdelta)*(ghdelta*2)
				in
					(colOffset, len)
			) ddaResult

let buildCSR (gridHalfWidth: f64, delta: f64, numRows: i32)
						 ((ddaResult: [](i32, (i32, [](i16, i16, f64)))))
						  : ([]i32, []i32, []f64) =
		let (is, vs) =
			unzip (
				map
			  (\(row:i32, (resLen: i32, _)) ->
			  	(row, resLen)
			  )	ddaResult
		  )
		in
		let lengths = scatter (replicate numRows 0) is vs in
	  let IA = scan (+) 0 (concat [0] lengths)
	  let (JA, A) =
	  unzip(
	    filter (\(_,len) -> len != -1.0) (
		    flatten(
		  		map (translateCSR (gridHalfWidth/delta)) ddaResult
		  	)
	  	)
	  )

	  in
	  (IA, JA, A)




-- inMode determines return format. 0 = COO, 1 = CSR
let main(detectorPositions: []f64, angles: []f64,
		 gridHalfWidth: f64, delta: f64, rowOffset: i32, inMode: i16, numRows: i32) :
		 (([]i32, []i32, []f64)) =

	-- Applies buildDDAParams partially with the arguments shared
	-- by all DDA instances.
	-- initFun is applied to each line, constructing the arguments needed
	-- by the DDA to run for that particular line.
	let initFun = buildDDAParams gridHalfWidth delta in
	--let n = f64.to_i32 ((gridHalfWidth/delta)*2.0*2.0) in
	let nFun = (\(angle) -> 2.0*gridHalfWidth/delta +
													f64.ceil(2.0*gridHalfWidth/delta*tan(toRad (toAxisAngle angle)))) in
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

		if inMode == 1i16 then
		  map initFun x |>
		  filter (\(_,_,_,_,_,disc,_) -> !disc) |>
		  map ddaFun |>
		  buildCSR (gridHalfWidth, delta, numRows)
		else
			unzip3
		  (
			  map initFun x |>
			  filter (\(_,_,_,_,_,disc,_) -> !disc) |>
			  map ddaFun |>
			  map (translateCOO (gridHalfWidth/delta)) |>
			  flatten |>
			  filter (\(_,_,len) -> len != -1.0)
		  )



-- parts must divide |angles|!!
-- Call this entry point if you dont have enough GPU memory to run it all in one go.
entry inParts(detectorPositions: []f64, angles: []f64,
		 gridHalfWidth: f64, delta: f64, parts: i32) :
		 ([]i32, []i32, []f64) =
		 let anglesPrRound = length(angles)/parts in
		 -- alloker på forhånd med replicate og brug in place opdateringer
		 let result = ([]i32, []i32, []f64) in
		 let (_, result, _) =
			 loop (i, (rowsRes, colsRes, distsRes), rowOffset) = (0, result, 0) for i < parts do
			    let numRows = length(detectorPositions)*(anglesPrRound*(i+1)-anglesPrRound-1)
			 		let (rows, cols, dists)
			 			= main(detectorPositions, angles[anglesPrRound*i:anglesPrRound*(i+1)],
										 gridHalfWidth, delta, rowOffset, 0i16, numRows)
			 	 in
			 	 (i+1, (concat rowsRes rows, concat colsRes cols, concat distsRes dists),
			 	 				rows[length(rows)-1]+1)
		  in result


entry calculateCSR(detectorPositions: []f64, angles: []f64,
		 gridHalfWidth: f64, delta: f64, rowOffset: i32, numRows: i32) :
		 (([]i32, []i32, []f64)) =
	main(detectorPositions, angles, gridHalfWidth, delta, rowOffset, 1i16, numRows)


entry calculateCOO(detectorPositions: []f64, angles: []f64,
		 gridHalfWidth: f64, delta: f64, rowOffset: i32) :
		 (([]i32, []i32, []f64)) =

  let numRows = length(detectorPositions)*length(angles) in
	main(detectorPositions, angles, gridHalfWidth, delta, rowOffset, 0i16, numRows)
