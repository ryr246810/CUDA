
// Copyright (C) 2007-2015  CEA/DEN, EDF R&D, OPEN CASCADE
//
// Copyright (C) 2003-2007  OPEN CASCADE, EADS/CCR, LIP6, CEA/DEN,
// CEDRAT, EDF R&D, LEG, PRINCIPIA R&D, BUREAU VERITAS
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
//
// See http://www.salome-platform.org/ or email : webmaster.salome@opencascade.com
//

#include <Standard_Stream.hxx>

#include <GEOMUtils.hxx>


// OCCT Includes
#include <BRepMesh_IncrementalMesh.hxx>

#include <BRepExtrema_DistShapeShape.hxx>

#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <BRepBndLib.hxx>
#include <BRepGProp.hxx>
#include <BRepTools.hxx>

#include <BRepClass3d_SolidClassifier.hxx>

#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_Sewing.hxx>

#include <BRepCheck_Analyzer.hxx>

#include <Bnd_Box.hxx>

#include <BOPTools_AlgoTools.hxx>

#include <TopAbs.hxx>
#include <TopExp.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Vertex.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Iterator.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_MapOfShape.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopTools_ListIteratorOfListOfShape.hxx>
#include <TopTools_Array1OfShape.hxx>

#include <Geom_Circle.hxx>
#include <Geom_Surface.hxx>
#include <Geom_Plane.hxx>
#include <Geom_SphericalSurface.hxx>
#include <Geom_ToroidalSurface.hxx>
#include <Geom_RectangularTrimmedSurface.hxx>

#include <GeomLProp_CLProps.hxx>
#include <GeomLProp_SLProps.hxx>

#include <GProp_GProps.hxx>
#include <GProp_PrincipalProps.hxx>

#include <TColStd_Array1OfReal.hxx>

#include <gp_Pln.hxx>
#include <gp_Lin.hxx>

#include <ShapeAnalysis.hxx>
#include <ShapeFix_Shape.hxx>
#include <ShapeFix_ShapeTolerance.hxx>

#include <ProjLib.hxx>
#include <ElSLib.hxx>

#include <vector>
#include <sstream>
#include <algorithm>

#include <Standard_Failure.hxx>
#include <Standard_NullObject.hxx>
#include <Standard_ErrorHandler.hxx> // CAREFUL ! position of this file is critic : see Lucien PIGNOLONI / OCC

#include <BOPDS_DS.hxx>
#include <BOPAlgo_CheckerSI.hxx>
#include <BOPCol_ListOfShape.hxx>
#include <TColStd_IndexedDataMapOfTransientTransient.hxx>

#include <StdFail_NotDone.hxx>

#include <BlockFix_BlockFixAPI.hxx>
#include <ShHealOper_ShapeProcess.hxx>
#include <TNaming_CopyShape.hxx>


#define BOP_SELF_INTERSECTIONS_LEVEL 4



void GEOMUtils::FixShapes(TopoDS_Shape& theInputShape)
{
  if (theInputShape.IsNull()) {
    Standard_NullObject::Raise("Null Shape given");
  }
  
  // Copy shape to avoid problems (Mantis issue 0021683)
  TopoDS_Shape aShapeCopy;
  TColStd_IndexedDataMapOfTransientTransient aMapTShapes;
  TNaming_CopyShape::CopyTool(theInputShape, aMapTShapes, aShapeCopy);
  theInputShape = aShapeCopy;
  
  // Repair result
  BRepCheck_Analyzer ana (aShapeCopy, false);
  if (!ana.IsValid()) {
    TopoDS_Shape aFixed;
    ShHealOper_ShapeProcess aHealer;
    aHealer.Perform(aShapeCopy, aFixed);
    if (aHealer.isDone())
      theInputShape = aFixed;
  }
}




TopoDS_Shape GEOMUtils::RemoveExtraEdges(const TopoDS_Shape &theShape)
{
  TopoDS_Shape aResult;
  
  if(!theShape.IsNull()) {
     BlockFix_BlockFixAPI aTool;
    
    aTool.OptimumNbFaces() = 0;
    aTool.SetShape(theShape);
    aTool.Perform();
    TopoDS_Shape aShape = aTool.Shape();
    
    if(CheckShape(aShape)) {
      aResult = aShape;
    }else{
      TopoDS_Shape aFixed;
      ShHealOper_ShapeProcess aHealer;
      aHealer.Perform(aResult, aFixed);
      if(aHealer.isDone() && CheckShape(aFixed))
	aResult = aFixed;
    }
  }
  
  return aResult;
}


/*
TopoDS_Shape GEOMUtils::RemoveExtraEdges(const TopoDS_Shape &theShape)
{
  TopoDS_Shape aShapeCopy;
  TColStd_IndexedDataMapOfTransientTransient aMapTShapes;
  TNaming_CopyShape::CopyTool(theShape, aMapTShapes, aShapeCopy);
  TopoDS_Shape aBlockOrComp = aShapeCopy;
  
  // 1. Improve solids with seam and/or degenerated edges
  BlockFix_BlockFixAPI aTool;
  //aTool.Tolerance() = toler;
  aTool.OptimumNbFaces() = 0;
  aTool.SetShape(aBlockOrComp);
  aTool.Perform();
  
  TopoDS_Shape aFixedExtra = aTool.Shape();
  
  // Repair result
  BRepCheck_Analyzer ana (aFixedExtra, false);
  if (!ana.IsValid()) {
    TopoDS_Shape aFixed;
    ShHealOper_ShapeProcess aHealer;
    aHealer.Perform(aFixedExtra, aFixed);
    if (aHealer.isDone())
      aFixedExtra = aFixed;
  }

  return aFixedExtra;
}

//*/







void GEOMUtils::FixShapeAfterBooleanOperation(TopoDS_Shape &aResult)
{
  TopTools_ListOfShape listShapeRes;
  AddSimpleShapes(aResult, listShapeRes);
  if (listShapeRes.Extent() == 1) {
    aResult = listShapeRes.First();
    if (aResult.IsNull()) return;
  }else{
    cout<<"GEOMUtils::FixShapeAfterBooleanOperation---------listShapeRes.Extent() = "<<listShapeRes.Extent()<<endl;
    return;
  }
}



void GEOMUtils::FixShapeAndRemoveExtraEdgesAfterBooleanOperation(TopoDS_Shape &aResult)
{
  TopTools_ListOfShape listShapeRes;
  AddSimpleShapes(aResult, listShapeRes);
  if (listShapeRes.Extent() == 1) {
    aResult = listShapeRes.First();
    if (aResult.IsNull()) return;
  }else{
    cout<<"GEOMUtils::FixShapeAfterBooleanOperation---------listShapeRes.Extent() = "<<listShapeRes.Extent()<<endl;
    return;
  }

  aResult = RemoveExtraEdges(aResult);
}



void GEOMUtils::CheckSI(TopoDS_Shape& aShape)
{
  BOPAlgo_CheckerSI aCSI;  // checker of self-interferences
  
  aCSI.SetLevelOfCheck(BOP_SELF_INTERSECTIONS_LEVEL);
  BOPCol_ListOfShape aList1;
  aList1.Append(aShape);
  aCSI.SetArguments(aList1);
  aCSI.Perform();
  /*
  if (aCSI.ErrorStatus() || aCSI.DS().Interferences().Extent() > 0) {
    StdFail_NotDone::Raise("Boolean operation will not be performed, because argument shape is self-intersected");
  }
  //*/
  if (aCSI.HasErrors() || aCSI.DS().Interferences().Extent() > 0) {
    StdFail_NotDone::Raise("Boolean operation will not be performed, because argument shape is self-intersected");
  }
}

