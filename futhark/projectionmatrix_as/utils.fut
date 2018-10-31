let toRad (angle: f64) : f64 =
	f64.pi/180.0*angle

let tan (angle: f64) : f64 =
	f64.sin(angle)/f64.cos(angle)

let toFloat (input: i32) : f64 =
	if input == 0 then
		0.0
	else
		f64.from_fraction input 1

let toAxisAngle (angle: f64) =
	if 0.0 <= angle && angle <= 45.0 then
		angle
	else
	if 45.0 < angle && angle < 90.0 then
		90.0-angle
	else if 90.0 <= angle && angle <= 135.0 then
		angle - 90.0
	else if 135.0 < angle && angle <= 180.0 then
		180.0-angle
	else if 180.0 < angle && angle <= 225.0 then
		angle-180.0
	else if 225.0 < angle && angle <= 270.0 then
		270.0-angle
	else if 270.0 < angle && angle <= 315.0 then
		angle-270.0
	else
		360.0-angle
