let main [n] (sane:[n]f32) (test:[n]f32) : bool =
  let std_dev = 0.0000005f32
  let bools = map2 (\s t -> if f32.abs(s-t) < std_dev then true else false) sane test
  in reduce (&&) true bools
