
let TEST (angles : []f32)
  (numrhos : i32): ([]bool, []f32, []f32) =
  let rep_cos = flatten <| map(\a-> replicate numrhos (f32.abs((f32.cos a))))(angles)
  let rep_sin = flatten <| map(\a-> replicate numrhos (f32.abs((f32.sin a))))(angles)
  let sin_greater_equal_cos = flatten <| map(\a-> replicate numrhos (f32.sin(a) >= f32.cos(a)))(angles)

  in (sin_greater_equal_cos,rep_sin, rep_cos)

let main  (angles : []f32)
           (numrhos : i32) :([]bool, []f32, []f32) =
           TEST angles numrhos
