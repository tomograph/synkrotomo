import "utils"

let handleFirstQuadrant(bottomIntersect: (f64, f64), rightIntersect: (f64, f64), 
												gridHalfWidth: f64, delta: f64, tanTheta: f64, b: f64,
												xDominant: bool) 
												: (f64, f64, f64, f64, bool, bool) =
	
	let (x,_) = bottomIntersect in
	let (x,y) =
		if x > gridHalfWidth || x < -gridHalfWidth then
			-- Intersects bottom outside grid, try right side
			let (_,y) = rightIntersect in
			if y > gridHalfWidth || y < -gridHalfWidth then
				-- Does not intersect bottom or rightside, line is outside grid.
				(0.0, 0.0) -- Impossible intersection
			else 
				rightIntersect
		else
			bottomIntersect
	in

	if (x,y) == (0.0, 0.0) then 
		-- Discard.
		(0.0, 0.0, 0.0, 0.0, true, true)
	else
		if xDominant then
			let (x, y, inBounds) = 
				if (x,y) == bottomIntersect then
					(f64.ceil(x/delta), 
					 tanTheta*f64.ceil(x/delta)+b/delta, 
					 false
					)
				else
					(x/delta, y/delta, true)
			in 
				(x, y, f64.round(x), f64.floor(y), inBounds, false)
		else
			let (x, y, inBounds) = 
				if (x,y) == rightIntersect then
					-- (x := y/delta/tanTheta, y)
					(((f64.floor(y/delta)-b/delta)/tanTheta), 
						f64.floor(y/delta), false)
				else
					(x/delta, y/delta, true)
			in 
				(x, y, f64.ceil(x), f64.round(y), inBounds, false)

let handleSecondQuadrant(topIntersect: (f64, f64), rightIntersect: (f64, f64), 
												gridHalfWidth: f64, delta: f64, tanTheta: f64, b: f64,
												xDominant: bool) 
												: (f64, f64, f64, f64, bool, bool) =		
	let (x,_) = topIntersect in
	let (x,y) =
		if x > gridHalfWidth || x < -gridHalfWidth then
			-- Intersects bottom outside grid, try right side
			let (_,y) = rightIntersect in
			if y > gridHalfWidth || y < -gridHalfWidth then
				-- Does not intersect bottom or rightside, line is outside grid.
				(0.0, 0.0) -- Impossible intersection
			else 
				rightIntersect
		else
			topIntersect
	in

	if (x,y) == (0.0, 0.0) then 
		-- Discard.
		(0.0, 0.0, 0.0, 0.0, true, true)
	else
		if xDominant then
			let (x, y, inBounds) = 
				if (x,y) == topIntersect then
					(f64.ceil(x/delta), 
					 tanTheta*f64.ceil(x/delta)+b/delta, 
					 false
					)
				else
					(x/delta, y/delta, true)
			in 
				(x, y, f64.round(x), f64.ceil(y), inBounds, false)
		else
			let (x, y, inBounds) = 
				if (x,y) == rightIntersect then
					-- (x := y/delta/tanTheta, y)
					(((f64.ceil(y/delta)-b/delta)/tanTheta), 
						f64.ceil(y/delta), false)
				else
					(x/delta, y/delta, true)
			in 
				(x, y, f64.ceil(x), f64.round(y), inBounds, false)	

let handleThirdQuadrant(topIntersect: (f64, f64), leftIntersect: (f64, f64), 
												gridHalfWidth: f64, delta: f64, tanTheta: f64, b: f64,
												xDominant: bool) 
												: (f64, f64, f64, f64, bool, bool) =		
	let (x,_) = topIntersect in
	let (x,y) =
		if x > gridHalfWidth || x < -gridHalfWidth then
			-- Intersects bottom outside grid, try right side
			let (_,y) = leftIntersect in
			if y > gridHalfWidth || y < -gridHalfWidth then
				-- Does not intersect top or leftside, line is outside grid.
				(0.0, 0.0) -- Impossible intersection
			else 
				leftIntersect
		else
			topIntersect
	in

	if (x,y) == (0.0, 0.0) then 
		-- Discard.
		(0.0, 0.0, 0.0, 0.0, true, true)
	else
		if xDominant then
			let (x, y, inBounds) = 
				if (x,y) == topIntersect then
					(f64.floor(x/delta), 
					 tanTheta*f64.floor(x/delta)+b/delta, 
					 false
					)
				else
					(x/delta, y/delta, true)
			in 
				(x, y, f64.round(x), f64.ceil(y), inBounds, false)
		else
			let (x, y, inBounds) = 
				if (x,y) == leftIntersect then
					-- (x := y/delta/tanTheta, y)
					(((f64.ceil(y/delta)-b/delta)/tanTheta), 
						f64.ceil(y/delta), false)
				else
					(x/delta, y/delta, true)
			in 
				(x, y, f64.floor(x), f64.round(y), inBounds, false)	

let handleFourthQuadrant(bottomIntersect: (f64, f64), leftIntersect: (f64, f64), 
												gridHalfWidth: f64, delta: f64, tanTheta: f64, b: f64,
												xDominant: bool)
												: (f64, f64, f64, f64, bool, bool) =		
	let (x,_) = bottomIntersect in
	let (x,y) =
		if x > gridHalfWidth || x < -gridHalfWidth then
			-- Intersects bottom outside grid, try right side
			let (_,y) = leftIntersect in
			if y > gridHalfWidth || y < -gridHalfWidth then
				-- Does not intersect top or leftside, line is outside grid.
				(0.0, 0.0) -- Impossible intersection
			else 
				leftIntersect
		else
			bottomIntersect
	in

	if (x,y) == (0.0, 0.0) then 
		-- Discard.
		(0.0, 0.0, 0.0, 0.0, true, true)
	else
		if xDominant then
			let (x, y, inBounds) = 
				if (x,y) == bottomIntersect then
					(f64.floor(x/delta), 
					 tanTheta*f64.floor(x/delta)+b/delta, 
					 false
					)
				else
					(x/delta, y/delta, true)
			in 
				(x, y, f64.round(x), f64.floor(y), inBounds, false)
		else
			let (x, y, inBounds) = 
				if (x,y) == leftIntersect then
					-- (x := y/delta/tanTheta, y)
					(((f64.floor(y/delta)-b/delta)/tanTheta), 
						f64.floor(y/delta), false)
				else
					(x/delta, y/delta, true)
			in 
				(x, y, f64.floor(x), f64.round(y), inBounds, false)	

entry getGridInterSection (gridHalfWidth: f64, delta: f64, detX: f64, detY: f64, 
													 angle: f64) : (f64, f64, f64, bool, bool) =

	
	let isYParallel = angle == 0.0  || angle == 180.0
	let isXParallel = angle == 90.0 || angle == 270.0
	in

	-- Handle the special case where the line is axis parallel
	if isYParallel then
		if !(detX >= -gridHalfWidth && detX < gridHalfWidth) then
			-- Outside grid, discard.
			(0.0, 0.0, 0.0, false, true)
		else if angle == 0.0 then
			-- Intersects bottom side
			(f64.floor(detX/delta), f64.floor(-gridHalfWidth/delta), 0.00, true, false)
		else
			-- Intersects top side
			(f64.floor(detX/delta), f64.floor(gridHalfWidth/delta), 0.00, true, false)
	else if isXParallel then
		if !(detY >= -gridHalfWidth && detY < gridHalfWidth) then
			-- Outside grid, discard.
			(0.0, 0.00, 0.00, false, true)
		else if angle == 90.0 then
			-- Intersects right side
			(f64.floor(gridHalfWidth/delta), f64.floor(detY/delta), 0.00, true, false)
		else
			-- Intersects left side
			(f64.floor(-gridHalfWidth/delta), f64.floor(detY/delta), 0.00, true, false)
	else
		-- Not x parallel or y parallel
		-- Add 270 to the line angle because when the detector angle is 0
		-- it's at angle 270 relative to the x-axis		
		let tanTheta = tan(toRad(angle+270.0))	
		let b = detY - detX*tanTheta -- y intersect of line
		-- Intersections with the line and each of the 4 sides of the grid
		let topIntersect = ((gridHalfWidth-b)/tanTheta, gridHalfWidth)
		let bottomIntersect = ((-gridHalfWidth-b)/tanTheta, -gridHalfWidth)
		let rightIntersect = (gridHalfWidth, tanTheta*gridHalfWidth+b)
		let leftIntersect = (-gridHalfWidth, tanTheta*(-gridHalfWidth)+b)

		let xDominant = (45.0 < angle && angle <= 135.0) || (225.0 <= angle && angle <= 315.0)
		

		-- Check each angle quadrant and initialize accordingly
		let (x, y, curX, curY, inBounds, discard) =
			if 0.0 < angle && angle < 90.0 then

				handleFirstQuadrant(bottomIntersect, rightIntersect, 
														gridHalfWidth, delta, 
														tanTheta, b, xDominant)

			else if 90.0 < angle && angle < 180.0 then

				handleSecondQuadrant(topIntersect, rightIntersect, 
														gridHalfWidth, delta, 
														tanTheta, b, xDominant)

			else if 180.0 < angle && angle < 270.0 then

				handleThirdQuadrant(topIntersect, leftIntersect, 
														gridHalfWidth, delta, 
														tanTheta, b, xDominant)
			else 

				handleFourthQuadrant(bottomIntersect, leftIntersect, 
														gridHalfWidth, delta, 
														tanTheta, b, xDominant)

		let chi = f64.sqrt((curX-x) ** 2.0 + (curY-y) ** 2.0)*delta
		in 
			(curX, curY, chi, inBounds, discard)


	
