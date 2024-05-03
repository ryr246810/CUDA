#Cmake file for library OCAFTOOL.so
#Author:   bys
#Created:  2021-10-10
#Modified: 2022-11-28

cmake_minimum_required(VERSION 3.9)

project(OCAFTOOL)

message(STATUS "Start to make OCAFTool")

set(
    lib_dir
    ./include
    ${CEMPIC_OCCT_INC}
    # ${CEMPIC_FREETYPE_INC}
)

include_directories(${lib_dir})

set(
    source_list
    ./BlockFix/BlockFix_BlockFixAPI.cxx
    ./BlockFix/BlockFix_SphereSpaceModifier.cxx
    ./BlockFix/BlockFix_CheckTool.cxx
    ./BlockFix/BlockFix_UnionEdges.cxx
    ./BlockFix/BlockFix.cxx
    ./BlockFix/BlockFix_UnionFaces.cxx
    ./BlockFix/BlockFix_PeriodicSurfaceModifier.cxx

    ./BREPNaming/BRepNaming.cxx
    ./BREPNaming/BRepNaming_TopNaming.cxx
    ./BREPNaming/BRepNaming_LoaderParent.cxx
    ./BREPNaming/BRepNaming_Loader.cxx
    ./BREPNaming/BRepImport/BRepNaming_ImportShape.cxx
    ./BREPNaming/Selection/BRepNaming_Selection.cxx
    ./BREPNaming/Boolean/BRepNaming_BooleanOperationFeat.cxx
    ./BREPNaming/Boolean/BRepNaming_Common.cxx
    ./BREPNaming/Boolean/BRepNaming_Fuse.cxx
    ./BREPNaming/Boolean/BRepNaming_MultiFuse.cxx
    ./BREPNaming/Boolean/BRepNaming_Cut.cxx
    ./BREPNaming/Boolean/BRepNaming_MultiCut.cxx
    ./BREPNaming/Transform/BRepNaming_Translate.cxx
    ./BREPNaming/Transform/BRepNaming_Rotate.cxx
    ./BREPNaming/Transform/BRepNaming_Mirror.cxx
    ./BREPNaming/Box/BRepNaming_Box.cxx
    ./BREPNaming/Cylinder/BRepNaming_Cylinder.cxx
    ./BREPNaming/Sphere/BRepNaming_Sphere.cxx
    ./BREPNaming/Cone/BRepNaming_Cone.cxx
    ./BREPNaming/Revolution/BRepNaming_Revolution.cxx
    ./BREPNaming/Prism/BRepNaming_Prism.cxx
    ./BREPNaming/Torus/BRepNaming_Torus.cxx
    ./BREPNaming/Circle/BRepNaming_Circle.cxx
    ./BREPNaming/Ellipse/BRepNaming_Ellipse.cxx
    ./BREPNaming/Parabola/BRepNaming_Parabola.cxx
    ./BREPNaming/Arc/BRepNaming_Arc.cxx
    ./BREPNaming/Vector/BRepNaming_Vector.cxx
    ./BREPNaming/Vertex/BRepNaming_Vertex.cxx
    ./BREPNaming/Edge/BRepNaming_Edge.cxx
    ./BREPNaming/Wire/BRepNaming_Wire.cxx
    ./BREPNaming/Face/BRepNaming_Face.cxx
    ./BREPNaming/Shell/BRepNaming_Shell.cxx
    ./BREPNaming/Solid/BRepNaming_Solid.cxx
    ./BREPNaming/ThruSections/BRepNaming_ThruSections.cxx
    ./BREPNaming/PipeShell/BRepNaming_PipeShell.cxx
    ./BREPNaming/Polygon/BRepNaming_Polygon.cxx
    ./BREPNaming/Pipe/BRepNaming_Pipe.cxx
    ./BREPNaming/CosPeriodEdge/BRepNaming_CosPeriodEdge.cxx
    ./BREPNaming/RecPeriodEdge/BRepNaming_RecPeriodEdge.cxx
    ./BREPNaming/HelixEdge/BRepNaming_HelixEdge.cxx    
    ./BREPNaming/PeriodShape/BRepNaming_PeriodShape.cxx    
    ./BREPNaming/MultiRotate/BRepNaming_MultiRotate.cxx    
    ./BREPNaming/Curve/BRepNaming_Curve.cxx

    ./GEOMUtils/GEOMUtils.cxx
    ./GEOMUtils/GEOMUtils_1.cxx

    ./OCAF/OCAF_ColorMap.cxx
    ./OCAF/OCAF_Driver.cxx
    ./OCAF/OCAF_IFunction.cxx
    ./OCAF/OCAF_IFunction_0.cxx
    ./OCAF/OCAF_IFunction_1.cxx
    ./OCAF/OCAF_IFunction_2.cxx
    ./OCAF/OCAF_IFunction_3.cxx
    ./OCAF/OCAF_Object.cxx
    ./OCAF/OCAF_Object_0.cxx
    ./OCAF/OCAF_ObjectTool.cxx
    ./OCAF/OCAF_IDisplayer.cxx
    ./OCAF/OCAF_ApplicationBase.cxx
    ./OCAF/OCAF_Application.cxx
    ./OCAF/OCAF_SolverEx.cxx
    ./OCAF/AISShape/OCAF_AISShape.cxx
    ./OCAF/DisplayShapeDriver/OCAF_AISFunctionDriver.cxx
    ./OCAF/BRepImport/OCAF_BRepImportDriver.cxx
    ./OCAF/BRepImport/OCAF_IBRepImport.cxx
    ./OCAF/STEPImport/OCAF_STEPImportDriver.cxx
    ./OCAF/STEPImport/OCAF_ISTEPImport.cxx
    ./OCAF/Selection/OCAF_ISelection.cxx
    ./OCAF/Selection/OCAF_SelectionDriver.cxx
    ./OCAF/Boolean/OCAF_IBooleanOperation.cxx
    ./OCAF/Boolean/OCAF_CommonDriver.cxx
    ./OCAF/Boolean/OCAF_ICommon.cxx
    ./OCAF/Boolean/OCAF_FuseDriver.cxx
    ./OCAF/Boolean/OCAF_IFuse.cxx
    ./OCAF/Boolean/OCAF_MultiFuseDriver.cxx
    ./OCAF/Boolean/OCAF_IMultiFuse.cxx
    ./OCAF/Boolean/OCAF_CutDriver.cxx
    ./OCAF/Boolean/OCAF_ICut.cxx
    ./OCAF/Boolean/OCAF_MultiCutDriver.cxx
    ./OCAF/Boolean/OCAF_IMultiCut.cxx
    ./OCAF/Transform/OCAF_ITransformParent.cxx
    ./OCAF/Transform/OCAF_ITranslate.cxx
    ./OCAF/Transform/OCAF_TranslateDriver.cxx
    ./OCAF/Transform/OCAF_IRotate.cxx
    ./OCAF/Transform/OCAF_RotateDriver.cxx
    ./OCAF/Transform/OCAF_IMirror.cxx
    ./OCAF/Transform/OCAF_MirrorDriver.cxx
    ./OCAF/Box/OCAF_BoxDriver.cxx
    ./OCAF/Box/OCAF_IBox.cxx
    ./OCAF/Cylinder/OCAF_CylinderDriver.cxx
    ./OCAF/Cylinder/OCAF_ICylinder.cxx
    ./OCAF/Sphere/OCAF_SphereDriver.cxx
    ./OCAF/Sphere/OCAF_ISphere.cxx
    ./OCAF/Cone/OCAF_ConeDriver.cxx
    ./OCAF/Cone/OCAF_ICone.cxx
    ./OCAF/Torus/OCAF_TorusDriver.cxx
    ./OCAF/Torus/OCAF_ITorus.cxx
    ./OCAF/Revolution/OCAF_IRevolution.cxx
    ./OCAF/Revolution/OCAF_RevolutionDriver.cxx
    ./OCAF/Prism/OCAF_IPrism.cxx
    ./OCAF/Prism/OCAF_PrismDriver.cxx
    ./OCAF/Circle/OCAF_CircleDriver.cxx
    ./OCAF/Circle/OCAF_ICircle.cxx
    ./OCAF/Ellipse/OCAF_EllipseDriver.cxx
    ./OCAF/Ellipse/OCAF_IEllipse.cxx
    ./OCAF/Parabola/OCAF_ParabolaDriver.cxx
    ./OCAF/Parabola/OCAF_IParabola.cxx
    ./OCAF/Arc/OCAF_ArcDriver.cxx
    ./OCAF/Arc/OCAF_IArc.cxx
    ./OCAF/Vector/OCAF_IVector.cxx
    ./OCAF/Vector/OCAF_VectorDriver.cxx
    ./OCAF/Vertex/OCAF_IVertex.cxx
    ./OCAF/Vertex/OCAF_VertexDriver.cxx
    ./OCAF/Edge/OCAF_IEdge.cxx
    ./OCAF/Edge/OCAF_EdgeDriver.cxx
    ./OCAF/Wire/OCAF_IWire.cxx
    ./OCAF/Wire/OCAF_WireDriver.cxx
    ./OCAF/Face/OCAF_IFace.cxx
    ./OCAF/Face/OCAF_FaceDriver.cxx
    ./OCAF/Shell/OCAF_IShell.cxx
    ./OCAF/Shell/OCAF_ShellDriver.cxx
    ./OCAF/Solid/OCAF_ISolid.cxx
    ./OCAF/Solid/OCAF_SolidDriver.cxx
    ./OCAF/ThruSections/OCAF_ThruSectionsDriver.cxx
    ./OCAF/ThruSections/OCAF_IThruSections.cxx
    ./OCAF/PipeShell/OCAF_IPipeShell.cxx
    ./OCAF/PipeShell/OCAF_PipeShellDriver.cxx
    ./OCAF/Polygon/OCAF_IPolygon.cxx
    ./OCAF/Polygon/OCAF_PolygonDriver.cxx
    ./OCAF/Pipe/OCAF_IPipe.cxx
    ./OCAF/Pipe/OCAF_PipeDriver.cxx
    ./OCAF/CosPeriodEdge/OCAF_ICosPeriodEdge.cxx
    ./OCAF/CosPeriodEdge/OCAF_CosPeriodEdgeDriver.cxx
    ./OCAF/RecPeriodEdge/OCAF_IRecPeriodEdge.cxx
    ./OCAF/RecPeriodEdge/OCAF_RecPeriodEdgeDriver.cxx
    ./OCAF/HelixEdge/OCAF_IHelixEdge.cxx
    ./OCAF/HelixEdge/OCAF_HelixEdgeDriver.cxx
    ./OCAF/PeriodShape/OCAF_IPeriodShape.cxx
    ./OCAF/PeriodShape/OCAF_PeriodShapeDriver.cxx
    ./OCAF/MultiRotate/OCAF_IMultiRotate.cxx
    ./OCAF/MultiRotate/OCAF_MultiRotateDriver.cxx
    ./OCAF/Curve/OCAF_ICurve.cxx
    ./OCAF/Curve/OCAF_CurveDriver.cxx
    
    ./ShHealingOper/ShHealOper_ShapeProcess.cxx
    ./ShHealingOper/ShHealOper_Tool.cxx
)

link_directories(${CEMPIC_OCCT_LIB})
# link_directories(${CEMPIC_FREETYPE_LIB})

add_library(OCAFTool SHARED ${source_list})

target_link_libraries(OCAFTool "-Wl,--start-group" ${OCCLIBS_Draw} ${OCCLIBS_DataExchange} ${OCCLIBS_ApplicationFramework} ${OCCLIBS_Visualization}  
${OCCLIBS_ModelingAlgorithms} ${OCCLIBS_ModelingData} ${OCCLIBS_Foundation} "-Wl,--end-group")