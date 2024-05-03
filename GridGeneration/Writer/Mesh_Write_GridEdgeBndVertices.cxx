
#include <Mesh_Write.hxx>


/********************************************************************************************/
/*
 * Function: GetPlaneMesh
 * Property: aDir: normal dir;  aDividingFactor: dividing factor
 */
/********************************************************************************************/
void 
Mesh_Write::
WriteEdgeVertices()
{
   
  BRep_Builder builder;	
  TopoDS_Compound Comp;	
  builder.MakeCompound(Comp);

  Standard_Integer NDIM = 2;
  for(Standard_Integer dir = 0; dir<NDIM; dir++){
    Standard_Integer Dir1 = dir;
    Standard_Integer Dir2 = (dir+1)%2;

    vector<Standard_Size> scalarIndxVex;
    Standard_Size VIndx[2];

    VIndx[Dir1] = 0;
    for(Standard_Integer index=0;index<=m_Data->GetZRGrid()->GetDimension(Dir2);index++){
      VIndx[Dir2] = index;
      Standard_Size tmpIndx;
      m_Data->GetZRGrid()->FillVertexIndx(VIndx,tmpIndx);
      scalarIndxVex.push_back(tmpIndx);
    } 

    Standard_Size TheIndex;
    Standard_Size SIZE = m_Data->GetZRGrid()->GetVertexSize(Dir1);

    const map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> > * theData1 = m_Data->GetGridBndDatas()->GetEdgeBndVertexDataOf( (ZRGridLineDir)Dir1);
    map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> > ::const_iterator  it1;
    
    for(Standard_Size i=0; i<scalarIndxVex.size(); i++){
      it1 = theData1->find(scalarIndxVex[i]);
      if( it1 != theData1->end() ) {
	Standard_Size nb = (it1->second).size(); 
	for(Standard_Size j=0;j<nb;j++)  {
	  TheIndex = it1->first + (it1->second[j].m_Index)*SIZE;

	  Standard_Real aPnt[2];
	  m_Data->GetZRGrid()->GetCoord_From_VertexScalarIndx(TheIndex, aPnt);
	  Standard_Real TheStep = m_Data->GetZRGrid()->GetStep(Dir1, (it1->second[j]).m_Index);
	  Standard_Real TheFrac = Standard_Real(it1->second[j].m_Frac)/Standard_Real( m_Data->GetZRGrid()->GetResolutionRatio() );
	  aPnt[Dir1] = aPnt[Dir1] + TheStep*TheFrac;
	  
	  TxVector<Standard_Real> theXYZ;
	  m_Data->GetZRDefine()->Convert_ZR_to_XYZ(aPnt, theXYZ);
	    
	  BRepBuilderAPI_MakeVertex MV( gp_Pnt(theXYZ[0],theXYZ[1],theXYZ[2]) ); 
	  if (MV.IsDone())  builder.Add(Comp,MV.Vertex());
	}
      }
    }
  }


  ostringstream sstr;
  sstr<<"IntEdgeVertices";
  sstr<<".brep";
  string s=sstr.str();
  
  BRepTools::Write(Comp,s.c_str());
}
