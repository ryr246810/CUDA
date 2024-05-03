#include <Mesh_Write.hxx>



/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
Mesh_Write::Mesh_Write()
{
  m_Data = NULL;
}



/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
Mesh_Write::Mesh_Write(Grid_Generation* _Data)
{
  m_Data = _Data;
}



/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
Mesh_Write::~Mesh_Write()
{
}



/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
void Mesh_Write::WriteIntPnts(const ZRGridLineDir aDir)
{
  const map<Standard_Size,vector<EdgeBndPntData>,less<Standard_Size> > * theData = m_Data->GetEdgeBndPntOf(aDir);

  map<Standard_Size,vector<EdgeBndPntData>,less<Standard_Size> >::const_iterator  iter;

  BRep_Builder builder;	
  TopoDS_Compound Comp;	
  builder.MakeCompound(Comp);

  for(iter = theData->begin(); iter != theData->end(); iter++){
    Standard_Size nb = (iter->second).size();
    for(Standard_Size i=0; i<nb; i++){
      BRepBuilderAPI_MakeVertex MV( iter->second[i].ThePnt ); 
      if (MV.IsDone())  builder.Add(Comp,MV.Vertex());
    }
  }

  Standard_Integer TheDir = Standard_Integer(aDir);

  ostringstream sstr;
  sstr<<"IntPnts";
  sstr<<"_";
  sstr<<TheDir;
  sstr<<".brep";
  string s=sstr.str();

  BRepTools::Write(Comp, s.c_str());
}



/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
void Mesh_Write::WriteFaceBndPnts()
{
  const vector<FaceBndPntData>* theCornerPnts =  m_Data->GetFaceBndPnt();
  const vector<FaceBndPntData>& TheResult = *theCornerPnts;

  Standard_Size nb = TheResult.size();
  BRep_Builder builder;	
  TopoDS_Compound Comp;	
  builder.MakeCompound(Comp);
  
  for(Standard_Size i=0; i<nb; i++){
    TopoDS_Vertex theVertex = BRepBuilderAPI_MakeVertex(TheResult[i].ThePnt).Vertex();
    builder.Add(Comp,  theVertex);
    
#ifdef MESH_WRITE_DBG
    ostringstream sstr;
    sstr<<"FaceBndPnt";
    sstr1<<"_";
    sstr<<i;
    sstr<<".brep";
    string s=sstr.str();
    BRepTools::Write(theVertex, s.c_str());
#endif
  }

  ostringstream sstr1;
  sstr1<<"FaceBndPnts";
  sstr1<<".brep";
  string s1=sstr1.str();
  
  BRepTools::Write(Comp,s1.c_str());
}
