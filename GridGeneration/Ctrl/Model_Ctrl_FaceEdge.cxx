#include <Model_Ctrl.hxx>


void
Model_Ctrl::
FindNeighBourFacesOfOneFace(const Standard_Integer theFaceIndex, 
			    std::set<Standard_Integer>& theNBFaceIndices) const
{
  theNBFaceIndices.clear();

  if(m_FaceWithEdgeTool.IsBound (theFaceIndex)){
    const TColStd_ListOfInteger& tmpEdgeIndices =  m_FaceWithEdgeTool.Find(theFaceIndex);

    TColStd_ListIteratorOfListOfInteger eListIter;
    for( eListIter.Initialize(tmpEdgeIndices); eListIter.More(); eListIter.Next() ){
      Standard_Integer tmpEdgeIndex = eListIter.Value();

      if(m_EdgeWithFaceTool.IsBound(tmpEdgeIndex)){
	const TColStd_ListOfInteger& tmpFaceIndices =  m_EdgeWithFaceTool.Find(tmpEdgeIndex);
	TColStd_ListIteratorOfListOfInteger fListIter;
	for( fListIter.Initialize(tmpFaceIndices); fListIter.More(); fListIter.Next() ){
	  Standard_Integer tmpFaceIndex = fListIter.Value();
	  if(theFaceIndex!=tmpFaceIndex){
	    std::set<Standard_Integer>::iterator iter = theNBFaceIndices.find(tmpFaceIndex);
	    if(iter==theNBFaceIndices.end()){
	      theNBFaceIndices.insert(tmpFaceIndex);
	    }
	  }
	}
      }
    }
  }
}
