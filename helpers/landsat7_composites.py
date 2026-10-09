"""Computed dataset expressions for Landsat."""

import ee


class C2SRExpressions:
  """Computations for the merged, masked Landsat C2 SR collection."""

  def _expression(self) -> ee.ImageCollection:
    """Prepare the merged Landsat C2 SR collection."""
    l7 = (
        ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
        .filter('WRS_ROW < 122')  # Remove night-time images.
        .map(prep_c2sr_l4l5l7)
    )
    return l7


def prep_c2sr_l4l5l7(image: ee.Image) -> ee.ImageCollection:
  """Scale and mask L5-L7 C2 SR."""
  optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
  thermal_band = image.select('ST_B6').multiply(0.00341802).add(149.0)

  scaled = optical_bands.addBands(thermal_band).select(
      ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'ST_B6'],
      ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']
  )

  # Select cloud free land and water pixels.
  qa = image.select(['QA_PIXEL'])
  # Clear if bits 0-5 are zero.
  mask1 = qa.bitwiseAnd(int('111111', 2)).eq(0)
  # Good snow/shadow/cloud confidence if bits 8-13 are equal to 010101.
  mask2 = qa.rightShift(8).bitwiseAnd(int('111111', 2)).eq(int('010101', 2))

  # Remove pixels marked as saturated or out of range.
  mask3 = image.select('QA_RADSAT').eq(0)
  mask4 = optical_bands.reduce(ee.Reducer.min()).gt(0)
  mask5 = optical_bands.reduce(ee.Reducer.max()).lt(1)
  # Mark hazy pixels using empirical AOD threshold.
  mask6 = image.select('SR_ATMOS_OPACITY').unmask(-1).lt(300)
  # mask6 = image.select('SR_ATMOS_OPACITY').unmask(-1).lt(150)

  # Put the new bands back into the original image container and mask them.
  return (
      image.select()
      .addBands(scaled)
      .updateMask(mask1.And(mask2).And(mask3).And(mask4).And(mask5).And(mask6))
  )