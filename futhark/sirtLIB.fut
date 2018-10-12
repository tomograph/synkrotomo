import "futlib/array"

module sirtLIB = {
  let vecADD [vct_len] (vct1 : [vct_len]f32) (vct2 : [vct_len]f32) : [vct_len]f32 =
  map2 (+) vct1 vct2
}
