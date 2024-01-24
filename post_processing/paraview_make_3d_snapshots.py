# trace generated using paraview version 5.11.2
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
from pathlib import Path
from argparse import ArgumentParser

import numpy as np

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input", "-i", type=str, help="root directory")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    Path(args.input).with_name('3d_view').mkdir(parents=True, exist_ok=True)

    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    # create a new 'XML Rectilinear Grid Reader'
    a3D_maskvtr = XMLRectilinearGridReader(registrationName='3D_mask.vtr', FileName=[args.input])
    a3D_maskvtr.PointArrayStatus = ['mask']

    # Properties modified on a3D_maskvtr
    a3D_maskvtr.TimeArray = 'None'

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')

    # show data in view
    a3D_maskvtrDisplay = Show(a3D_maskvtr, renderView1, 'UniformGridRepresentation')

    # trace defaults for the display properties.
    a3D_maskvtrDisplay.Representation = 'Outline'
    a3D_maskvtrDisplay.ColorArrayName = [None, '']
    a3D_maskvtrDisplay.SelectTCoordArray = 'None'
    a3D_maskvtrDisplay.SelectNormalArray = 'None'
    a3D_maskvtrDisplay.SelectTangentArray = 'None'
    a3D_maskvtrDisplay.OSPRayScaleArray = 'mask'
    a3D_maskvtrDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    a3D_maskvtrDisplay.SelectOrientationVectors = 'None'
    a3D_maskvtrDisplay.ScaleFactor = 44.1
    a3D_maskvtrDisplay.SelectScaleArray = 'None'
    a3D_maskvtrDisplay.GlyphType = 'Arrow'
    a3D_maskvtrDisplay.GlyphTableIndexArray = 'None'
    a3D_maskvtrDisplay.GaussianRadius = 2.205
    a3D_maskvtrDisplay.SetScaleArray = ['POINTS', 'mask']
    a3D_maskvtrDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    a3D_maskvtrDisplay.OpacityArray = ['POINTS', 'mask']
    a3D_maskvtrDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    a3D_maskvtrDisplay.DataAxesGrid = 'GridAxesRepresentation'
    a3D_maskvtrDisplay.PolarAxes = 'PolarAxesRepresentation'
    a3D_maskvtrDisplay.ScalarOpacityUnitDistance = 2.1369457961418186
    a3D_maskvtrDisplay.OpacityArrayName = ['POINTS', 'mask']
    a3D_maskvtrDisplay.ColorArray2Name = ['POINTS', 'mask']
    a3D_maskvtrDisplay.SliceFunction = 'Plane'
    a3D_maskvtrDisplay.Slice = 65
    a3D_maskvtrDisplay.SelectInputVectors = [None, '']
    a3D_maskvtrDisplay.WriteLog = ''

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    a3D_maskvtrDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 2.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    a3D_maskvtrDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 2.0, 1.0, 0.5, 0.0]

    # init the 'Plane' selected for 'SliceFunction'
    a3D_maskvtrDisplay.SliceFunction.Origin = [220.5, 186.0, 65.5]

    # reset view to fit data
    renderView1.ResetCamera(False)

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # update the view to ensure updated data information
    renderView1.Update()

    # create a new 'Threshold'
    threshold1 = Threshold(registrationName='Threshold1', Input=a3D_maskvtr)
    threshold1.Scalars = ['POINTS', 'mask']
    threshold1.UpperThreshold = 2.0

    # Properties modified on threshold1
    threshold1.LowerThreshold = 1.0

    # show data in view
    threshold1Display = Show(threshold1, renderView1, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    threshold1Display.Representation = 'Surface'
    threshold1Display.ColorArrayName = [None, '']
    threshold1Display.SelectTCoordArray = 'None'
    threshold1Display.SelectNormalArray = 'None'
    threshold1Display.SelectTangentArray = 'None'
    threshold1Display.OSPRayScaleArray = 'mask'
    threshold1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    threshold1Display.SelectOrientationVectors = 'None'
    threshold1Display.ScaleFactor = 17.03863525390625
    threshold1Display.SelectScaleArray = 'None'
    threshold1Display.GlyphType = 'Arrow'
    threshold1Display.GlyphTableIndexArray = 'None'
    threshold1Display.GaussianRadius = 0.8519317626953126
    threshold1Display.SetScaleArray = ['POINTS', 'mask']
    threshold1Display.ScaleTransferFunction = 'PiecewiseFunction'
    threshold1Display.OpacityArray = ['POINTS', 'mask']
    threshold1Display.OpacityTransferFunction = 'PiecewiseFunction'
    threshold1Display.DataAxesGrid = 'GridAxesRepresentation'
    threshold1Display.PolarAxes = 'PolarAxesRepresentation'
    threshold1Display.ScalarOpacityUnitDistance = 6.325582709073908
    threshold1Display.OpacityArrayName = ['POINTS', 'mask']
    threshold1Display.SelectInputVectors = [None, '']
    threshold1Display.WriteLog = ''

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    threshold1Display.ScaleTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 2.0, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    threshold1Display.OpacityTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 2.0, 1.0, 0.5, 0.0]

    # update the view to ensure updated data information
    renderView1.Update()

    renderView1.ResetActiveCameraToPositiveY()

    # reset view to fit data
    renderView1.ResetCamera(False)

    # hide data in view
    Hide(a3D_maskvtr, renderView1)

    # Show orientation axes
    renderView1.OrientationAxesVisibility = 1

    # Hide orientation axes
    renderView1.OrientationAxesVisibility = 0

    renderView1.ApplyIsometricView()

    # reset view to fit data
    renderView1.ResetCamera(False)

    renderView1.ResetActiveCameraToPositiveY()

    # reset view to fit data
    renderView1.ResetCamera(False)

    # set scalar coloring
    ColorBy(threshold1Display, ('POINTS', 'mask'))

    # rescale color and/or opacity maps used to include current data range
    threshold1Display.RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    threshold1Display.SetScalarBarVisibility(renderView1, True)

    # get 2D transfer function for 'mask'
    maskTF2D = GetTransferFunction2D('mask')

    # get color transfer function/color map for 'mask'
    maskLUT = GetColorTransferFunction('mask')
    maskLUT.TransferFunction2D = maskTF2D
    maskLUT.RGBPoints = [1.0, 0.231373, 0.298039, 0.752941, 1.5, 0.865003, 0.865003, 0.865003, 2.0, 0.705882, 0.0156863, 0.14902]
    maskLUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'mask'
    maskPWF = GetOpacityTransferFunction('mask')
    maskPWF.Points = [1.0, 0.0, 0.5, 0.0, 2.0, 1.0, 0.5, 0.0]
    maskPWF.ScalarRangeInitialized = 1

    # hide color bar/color legend
    threshold1Display.SetScalarBarVisibility(renderView1, False)

    # Properties modified on threshold1Display
    threshold1Display.Opacity = 0.8

    # Properties modified on threshold1Display
    threshold1Display.Specular = 0.5

    # reset view to fit data
    renderView1.ResetCamera(False)

    # get animation scene
    animationScene1 = GetAnimationScene()

    # Properties modified on animationScene1
    animationScene1.NumberOfFrames = 120

    # get the time-keeper
    timeKeeper1 = GetTimeKeeper()

    # get camera animation track for the view
    cameraAnimationCue1 = GetCameraTrack(view=renderView1)

    # Update the pipeline to reflect the changes
    UpdatePipeline()

    # Get the bounding box of the thresholded data
    thresholded_bounds = threshold1.GetDataInformation().GetBounds()

    # Calculate the extents in the X, Y, and Z dimensions
    extent_x = thresholded_bounds[1] - thresholded_bounds[0]
    extent_y = thresholded_bounds[3] - thresholded_bounds[2]
    extent_z = thresholded_bounds[5] - thresholded_bounds[4]

    # Calculate the parallel scale as half of the maximum extent in X or Y
    parallel_scale = max(extent_x, extent_y, extent_z) * 0.5

    # Calculate the focal point (center of the bounding box)
    center_x = (thresholded_bounds[0] + thresholded_bounds[1]) / 2
    center_y = (thresholded_bounds[2] + thresholded_bounds[3]) / 2
    center_z = (thresholded_bounds[4] + thresholded_bounds[5]) / 2
    focal_point = [center_x, center_y, center_z]

    radius = parallel_scale * 5.0

    # Define the number of points on the circle
    num_points = 36 # For example, every 10 degrees
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    # Calculate position points on the circle
    position_path_points = []
    for angle in angles:
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        z = center_z # Keep the camera at the same z-level, or adjust if needed
        position_path_points += [x, y, z]

    # create a new key frame
    keyFrame30341 = CameraKeyFrame()
    keyFrame30341.Position = position_path_points[0:3]
    keyFrame30341.FocalPoint = focal_point
    keyFrame30341.ViewUp = [0.0, 0.0, 1.0]
    keyFrame30341.ParallelScale = parallel_scale

    # create a new key frame
    keyFrame30342 = CameraKeyFrame()
    keyFrame30342.KeyTime = 1.0
    keyFrame30342.Position = position_path_points[0:3]
    keyFrame30342.FocalPoint = focal_point
    keyFrame30342.ViewUp = [0.0, 0.0, 1.0]
    keyFrame30342.ParallelScale = parallel_scale

    # initialize the animation track
    keyFrame30341.PositionPathPoints = position_path_points
    keyFrame30341.FocalPathPoints = focal_point
    keyFrame30341.ClosedPositionPath = 1
    cameraAnimationCue1.Mode = 'Path-based'
    cameraAnimationCue1.KeyFrames = [keyFrame30341, keyFrame30342]

    # save animation
    root = Path(args.input).with_name('3d_view')
    filename = str(root / Path('view.png'))
    SaveAnimation(filename, renderView1, ImageResolution=[1600, 1600],
        FrameWindow=[0, 119], SuffixFormat='_%03d')