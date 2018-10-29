-- ==
-- input@../benchmarks/testinput_seq_500_5
-- input@../benchmarks/testinput_seq_500_10
-- input@../benchmarks/testinput_seq_500_15
-- input@../benchmarks/testinput_seq_500_20
-- input@../benchmarks/testinput_seq_500_25
-- input@../benchmarks/testinput_seq_500_30
-- input@../benchmarks/testinput_seq_500_35
-- input@../benchmarks/testinput_seq_500_40
-- input@../benchmarks/testinput_seq_500_45
-- input@../benchmarks/testinput_seq_500_50
-- input@../benchmarks/testinput_seq_500_55
module M = import "algorithm"
--this is the main function
let main(detectorPositions: []f64, angles: []f64,
		 gridHalfWidth: f64, delta: f64, rowOffset: i32, inMode: i16, numRows: i32) :
		 (([]i32, []i32, []f64)) =
		 M.main(detectorPositions, angles, gridHalfWidth, delta, rowOffset, inMode, numRows)
