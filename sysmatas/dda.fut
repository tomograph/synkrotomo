import "futlib/math"
import "futlib/array"
import "utils"

module DDA = {

let fixBounds (angle: f64, xstep: i16, ystep: i16, x: i16, y: i16,  primaryX: bool,
							 xBound: i16, yBound: i16) 
					: (f64, i16, i16, i16, i16, bool, i16, i16) =

					let (x, xBound) =
						if xstep < 0i16 then
							(x-1i16, -xBound-1i16)
						else
							(x, xBound)
					in
					let (y, yBound) =
						if ystep < 0i16 then
							(y-1i16, -yBound-1i16)
						else
							(y, yBound)
					in 
						(angle, xstep, ystep, x, y, primaryX, xBound, yBound)

-- Initialises the dda by converting the angle such that it is the angle
-- with the nearest axis, thus obeying 0 <= tan angle <= 1.
-- It also determines which way to step on both axes and whether 
-- x or y is the primary axis
let init (angle: f64, x: i16, y: i16, xBound: i16, yBound: i16) 
					: (f64, i16, i16, i16, i16, bool, i16, i16) =
	let (angle, xstep, ystep, primaryX) =
		if 0.0 <= angle && angle <= 45.0 then
			(angle, -1i16, 1i16, false)
		else
		if 45.0 < angle && angle < 90.0 then
			(90.0-angle, -1i16, 1i16, true)
		else if 90.0 <= angle && angle <= 135.0 then
			(angle - 90.0, -1i16, -1i16, true)
		else if 135.0 < angle && angle <= 180.0 then
			(180.0-angle, -1i16, -1i16, false)
		else if 180.0 < angle && angle <= 225.0 then
			(angle-180.0, 1i16, -1i16, false)
		else if 225.0 < angle && angle < 270.0 then
			(270.0-angle, 1i16, -1i16, true)
		else if angle == 270.0 then
			(0.0, 1i16, 1i16, true)
		else if 270.0 < angle && angle <= 315.0 then
			(angle-270.0, 1i16, 1i16, true)
		else
			(360.0-angle, 1i16, 1i16, false)
	in 
		fixBounds(angle, xstep, ystep, x, y, primaryX, xBound, yBound)

let parallelInit (angle: f64, xstep: i16, ystep: i16, x: i16, y: i16)
								: (i16, i16) =

		if (angle == 0.0 || angle == 180.0) && xstep < 0i16 then
			(x+1i16, y)
		else if (angle == 90.0 || angle == 270.0) && ystep < 0i16 then
			(x, y+1i16)
		else 
			(x,y)


let ddaX (x: i16, y: i16, xstep: i16, ystep: i16, 
					chi: f64, delta: f64, lambda: f64, tau: f64, xBound: i16, 
					yBound: i16,  i: i32, res: *[](i16, i16, f64), n: i32)
					: (i32, *[](i16, i16, f64))  =
		
		let notDone = true in
		let lastIndex = 0 in
		let (_, _, _, _, res , _, lastIndex) =
			loop (x, y, chi, i, res, notDone, lastIndex) while notDone do 
				if chi + tau < delta then 
					(x+xstep, y, chi+tau, i+1,
						res with [i] <- (x,y, lambda), 
						!(x+xstep == xBound), i)
				else
					if y + ystep == yBound then
						(x, y, chi, i, 
							res with [i] <- (x,y, ((delta-chi)/tau)*lambda ), false, i)

					else if x+xstep == xBound then
						(x, y, chi, i, 
							(res with [i] <- (x, y, ((delta-chi)/tau)*lambda) )
									 with [i+1] <- (x, y+ystep, ((tau-delta+chi)/tau)*lambda)
							, false, i+1)

					else
						(x+xstep, y+ystep, chi + tau - delta, i+2, 
							(res with [i]   <- (x, y, ((delta-chi)/tau)*lambda) )
									 with [i+1] <- (x, y+ystep, ((tau-delta+chi)/tau)*lambda)
							, !(x+xstep == xBound  || y+ystep == yBound), i+1)
		in
			(lastIndex+1, res)
			


let ddaY (x: i16, y: i16, xstep: i16, ystep: i16, 
					chi: f64, delta: f64, lambda: f64, tau: f64, xBound: i16, 
					yBound: i16,  i: i32, res: *[](i16, i16, f64), n: i32)
					: (i32, *[](i16, i16, f64))  =
		
		let notDone = true
		let lastIndex = 0 in
		let (_, _, _, _, res, _, lastIndex) =
			loop (x, y, chi, i, res, notDone, lastIndex) while notDone do 
				if chi + tau < delta then 
					(x, y+ystep, chi+tau, i+1,
						res with [i] <- (x, y, lambda), 
						!(y+ystep == yBound), i)
				else
					if x + xstep == xBound then
						(x, y, chi, i, 
							res with [i] <- (x, y, ((delta-chi)/tau)*lambda), false, i)

					else if y+ystep == yBound then
						(x, y, chi, i, 

							(res with [i] 	<- (x, y, ((delta-chi)/tau)*lambda)) 
									 with [i+1] <- (x+xstep, y, ((tau-delta+chi)/tau)*lambda )
							, false, i+1)

					else
						(x+xstep, y+ystep, chi + tau - delta, i+2, 
							(res with [i] 	<- (x, y, ((delta-chi)/tau)*lambda)) 
									 with [i+1] <- (x+xstep, y, ((tau-delta+chi)/tau)*lambda )
							, !(x+xstep == xBound  || y+ystep == yBound), i+1)
		in
			(lastIndex+1, res)





-- row is a passthru variable
-- discard is passed on to avoid an extra map 
let run (n:i32, xBound: i16, yBound: i16, delta: f64)
					(angleIn: f64, x: i16, y: i16, chi: f64, inBounds: bool,
					 _: bool,
					 row: i32)
					 
					: (i32, (i32, *[n](i16, i16, f64))) = 
	
	-- Initialize the dda
	let (angle, xstep, ystep, x, y, primaryX, xBound, yBound) 
		= init(angleIn, x, y, xBound, yBound)
	-- Fix starting conditions for axis parallel lines
	let (x, y)
		= parallelInit(angleIn, xstep, ystep, x, y)

	let radAngle = toRad(angle)
	-- -1 is the stop value. After the DDA is run, the remainder of the dist 
	-- array should be discarded when -1.0 is encoun tered
	let res = replicate n (0i16, 0i16, -1.0)


	let tau = delta*tan(radAngle)
	let lambda = delta/f64.cos(radAngle)
	in
	if primaryX then 
	 if !inBounds then
		if chi+tau < delta then
			-- skip the first pixel
			(row, ddaX(x+xstep, y, xstep, ystep, chi+tau, 
						delta, lambda, tau,
						xBound, yBound, 0, res, n)
			)

		else 
			-- Step the dda once and record the second pixel
			(row, ddaX(x+xstep, y+ystep, xstep, ystep, chi + tau - delta, 
						delta, lambda, tau,
						xBound, yBound, 1,
						 res with [0] <- (x, y+ystep, ((tau-delta+chi)/tau)*lambda ),
					 n)
			)
	 else
		(row, ddaX(x, y, xstep, ystep, chi, 
					delta, lambda, tau,
					xBound, yBound, 0, res, n)
		)
				
	else
	 if !inBounds then
		if chi+tau < delta then
			-- skip the first pixel
			(row, ddaY(x, y+ystep, xstep, ystep, chi+tau, 
						delta, lambda, tau,
						xBound, yBound, 0, res, n))

		else 
			-- Step the dda once and record the second pixel
			(row, ddaY(x+xstep, y+ystep, xstep, ystep, chi + tau - delta, 
						delta, lambda, tau,
						xBound, yBound, 1, 
						res with [0] <- (x+xstep, y, ((tau-delta+chi)/tau)*lambda ),
					 n)
			)
	 else
		(row, ddaY(x, y, xstep, ystep, chi, 
					delta, lambda, tau,
					xBound, yBound, 0, res, n))


-- Non higher order entry, used for running test benchmarks
--entry main(n:i32, xBound: i32, yBound: i32, delta: f64, angleIn: f64, x: i32, y: i32, chi: f64, inBounds: bool) 
	--				: ([n](i32, i32, f64)) =
		--	run (n, xBound, yBound, delta) (angleIn, x, y, chi, inBounds)

}
