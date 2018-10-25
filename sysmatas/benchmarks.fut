-- == 
-- input@../benchmarks/testinput
-- input@../benchmarks/testinput_500_0
-- input@../benchmarks/testinput_750_0
-- input@../benchmarks/testinput_1000_0
-- input@../benchmarks/testinput_1250_0
-- input@../benchmarks/testinput_1500_0
-- input@../benchmarks/testinput_1750_0
-- input@../benchmarks/testinput_2000_0
-- input@../benchmarks/testinput_500_20
-- input@../benchmarks/testinput_750_20
-- input@../benchmarks/testinput_1000_20
-- input@../benchmarks/testinput_1250_20
-- input@../benchmarks/testinput_1500_20
-- input@../benchmarks/testinput_1750_20
-- input@../benchmarks/testinput_2000_20
-- input@../benchmarks/testinput_100_45
-- input@../benchmarks/testinput_200_45
-- input@../benchmarks/testinput_300_45
-- input@../benchmarks/testinput_400_45
-- input@../benchmarks/testinput_500_45
-- input@../benchmarks/testinput_750_45
-- input@../benchmarks/testinput_1000_45
-- input@../benchmarks/testinput_1250_45
-- input@../benchmarks/testinput_1500_45
-- input@../benchmarks/testinput_1750_45
-- input@../benchmarks/testinput_2000_45
module M = import "algorithm"

let main(detectorPositions: []f64, angles: []f64, 
		 gridHalfWidth: f64, delta: f64, rowOffset: i32, inMode: i16, numRows: i32) :
		 (([]i32, []i32, []f64)) =
		 M.main(detectorPositions, angles, gridHalfWidth, delta, rowOffset, inMode, numRows)