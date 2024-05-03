#include <Model_Ctrl.hxx>
#include <Adaptor3d_HSurfaceTool.hxx>

void 
Model_Ctrl::
AnalysisShape()
{

  bool isParallelX;
  bool isParallelY;
  bool isParallelZ;

  TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;
  for(Iter.Initialize(m_ShapesWithTypeTool); Iter.More(); Iter.Next() ){
    const TopoDS_Shape & theShape = Iter.Key();

    TopExp_Explorer Ex;
    for(Ex.Init(theShape,TopAbs_FACE); Ex.More(); Ex.Next()) { 
      TopoDS_Face currentface = TopoDS::Face(Ex.Current());
      BRepAdaptor_Surface surface;
      surface.Initialize(currentface, Standard_True);
      Handle(BRepAdaptor_HSurface) Hsurface = new BRepAdaptor_HSurface(surface);
      
      GeomAbs_SurfaceType surfaceType =Adaptor3d_HSurfaceTool::GetType(Hsurface);
      
      //cout<<"surfaceType = "<<surfaceType<<endl;

      if(surfaceType==GeomAbs_Plane){
	gp_Pln currPlane = Hsurface->Plane();
	const gp_Pnt& plnLoc = currPlane.Location();
	const gp_Ax1& plnAx = currPlane.Axis();
	const gp_Dir& plnDir = plnAx.Direction();

	if(IsOneCoordDirrection(plnDir, isParallelX, isParallelY, isParallelZ)){
	  if(isParallelX){
	    m_SpecialXDatas.insert(pair<Standard_Real, Standard_Integer>(plnLoc.X(), 0));
	  }else if(isParallelY){
	    m_SpecialYDatas.insert(pair<Standard_Real, Standard_Integer>(plnLoc.Y(), 0));
	  }else if(isParallelZ){
	    m_SpecialZDatas.insert(pair<Standard_Real, Standard_Integer>(plnLoc.Z(), 0));
	  }else{

	  }
	}
      }
      if(surfaceType==GeomAbs_SurfaceOfRevolution){
	Handle(Adaptor3d_HCurve) aBasisCurve = Adaptor3d_HSurfaceTool::BasisCurve(Hsurface);
	Standard_Real T0 = aBasisCurve->FirstParameter();
	Standard_Real T1 = aBasisCurve->LastParameter();

	gp_Ax1 aRevAx = Adaptor3d_HSurfaceTool::AxeOfRevolution(Hsurface);
	const gp_Dir aRevAxDir = aRevAx.Direction();

	if(IsOneCoordDirrection(aRevAxDir, isParallelX, isParallelY, isParallelZ)){
	  gp_Pnt P0 = aBasisCurve->Value(T0);
	  gp_Pnt P1 = aBasisCurve->Value(T1);

	  if(isParallelX){
	    m_SpecialXDatas.insert(pair<Standard_Real, Standard_Integer>(P0.X(), 0));
	    m_SpecialXDatas.insert(pair<Standard_Real, Standard_Integer>(P1.X(), 0));
	  }else if(isParallelY){
	    m_SpecialYDatas.insert(pair<Standard_Real, Standard_Integer>(P0.Y(), 0));
	    m_SpecialYDatas.insert(pair<Standard_Real, Standard_Integer>(P1.Y(), 0));
	  }else if(isParallelZ){
	    m_SpecialZDatas.insert(pair<Standard_Real, Standard_Integer>(P0.Z(), 0));
	    m_SpecialZDatas.insert(pair<Standard_Real, Standard_Integer>(P1.Z(), 0));
	  }else{

	  }

	}
      }

      if(surfaceType == GeomAbs_Cylinder){
	Standard_Real U0 = Adaptor3d_HSurfaceTool::FirstUParameter(Hsurface);
	Standard_Real V0 = Adaptor3d_HSurfaceTool::FirstVParameter(Hsurface);
	Standard_Real V1 = Adaptor3d_HSurfaceTool::LastVParameter(Hsurface);

	gp_Cylinder aCyl = Adaptor3d_HSurfaceTool::Cylinder(Hsurface);
	const gp_Ax1 aCylAx = aCyl.Axis(); 
	const gp_Dir aCylAxDir = aCylAx.Direction();

	if(IsOneCoordDirrection(aCylAxDir, isParallelX, isParallelY, isParallelZ)){
	  gp_Pnt P0 =  Adaptor3d_HSurfaceTool::Value(Hsurface, U0, V0);
 	  gp_Pnt P1 =  Adaptor3d_HSurfaceTool::Value(Hsurface, U0, V1);

	  if(isParallelX){
	    m_SpecialXDatas.insert(pair<Standard_Real, Standard_Integer>(P0.X(), 0));
	    m_SpecialXDatas.insert(pair<Standard_Real, Standard_Integer>(P1.X(), 0));
	  }else if(isParallelY){
	    m_SpecialYDatas.insert(pair<Standard_Real, Standard_Integer>(P0.Y(), 0));
	    m_SpecialYDatas.insert(pair<Standard_Real, Standard_Integer>(P1.Y(), 0));
	  }else if(isParallelZ){
	    m_SpecialZDatas.insert(pair<Standard_Real, Standard_Integer>(P0.Z(), 0));
	    m_SpecialZDatas.insert(pair<Standard_Real, Standard_Integer>(P1.Z(), 0));
	  }else{

	  }
	}
      }

      if(surfaceType == GeomAbs_Cone){
	Standard_Real U0 = Adaptor3d_HSurfaceTool::FirstUParameter(Hsurface);
	Standard_Real V0 = Adaptor3d_HSurfaceTool::FirstVParameter(Hsurface);
	Standard_Real V1 = Adaptor3d_HSurfaceTool::LastVParameter(Hsurface);

	gp_Cone aCone = Adaptor3d_HSurfaceTool::Cone(Hsurface);
	const gp_Ax1 aConeAx = aCone.Axis(); 
	const gp_Dir aConeAxDir = aConeAx.Direction();

	if(IsOneCoordDirrection(aConeAxDir, isParallelX, isParallelY, isParallelZ)){
	  gp_Pnt P0 =  Adaptor3d_HSurfaceTool::Value(Hsurface, U0, V0);
 	  gp_Pnt P1 =  Adaptor3d_HSurfaceTool::Value(Hsurface, U0, V1);

	  if(isParallelX){
	    m_SpecialXDatas.insert(pair<Standard_Real, Standard_Integer>(P0.X(), 0));
	    m_SpecialXDatas.insert(pair<Standard_Real, Standard_Integer>(P1.X(), 0));
	  }else if(isParallelY){
	    m_SpecialYDatas.insert(pair<Standard_Real, Standard_Integer>(P0.Y(), 0));
	    m_SpecialYDatas.insert(pair<Standard_Real, Standard_Integer>(P1.Y(), 0));
	  }else if(isParallelZ){
	    m_SpecialZDatas.insert(pair<Standard_Real, Standard_Integer>(P0.Z(), 0));
	    m_SpecialZDatas.insert(pair<Standard_Real, Standard_Integer>(P1.Z(), 0));
	  }else{

	  }
	}
      }
    }
  }


  m_SpecialXCoordVec.clear();
  map<Standard_Real, Standard_Integer, less<Standard_Real> >::iterator iter;
  Standard_Integer i = 0;
  for(iter=m_SpecialXDatas.begin(); iter!=m_SpecialXDatas.end(); iter++){
    m_SpecialXCoordVec.push_back(iter->first);
    iter->second = i;
    i++;
  }

  m_SpecialYCoordVec.clear();
  i = 0;
  for(iter=m_SpecialYDatas.begin(); iter!=m_SpecialYDatas.end(); iter++){
    m_SpecialYCoordVec.push_back(iter->first);
    iter->second = i;
    i++;
  }

  m_SpecialZCoordVec.clear();
  i = 0;
  for(iter=m_SpecialZDatas.begin(); iter!=m_SpecialZDatas.end(); iter++){
    m_SpecialZCoordVec.push_back(iter->first);
    iter->second = i;
    i++;
  }


  //*
  cout<<endl;
  cout<<"Model_Ctrl::AnalysisShape------------------------------------------>>>"<<endl;

  cout<<"m_SpecialXDatas = [ ";
  for(iter=m_SpecialXDatas.begin(); iter!=m_SpecialXDatas.end(); iter++){
    cout<<"(";
    cout<<iter->first;
    cout<<", ";
    cout<<iter->second;
    cout<<") ";
  }
  cout<<"]"<<endl;

  cout<<"m_SpecialYDatas = [ ";
  for(iter=m_SpecialYDatas.begin(); iter!=m_SpecialYDatas.end(); iter++){
    cout<<"(";
    cout<<iter->first;
    cout<<", ";
    cout<<iter->second;
    cout<<") ";
  }
  cout<<"]"<<endl;

  cout<<"m_SpecialZDatas = [ ";
  for(iter=m_SpecialZDatas.begin(); iter!=m_SpecialZDatas.end(); iter++){
    cout<<"(";
    cout<<iter->first;
    cout<<", ";
    cout<<iter->second;
    cout<<") ";
  }
  cout<<"]"<<endl;


  cout<<"Model_Ctrl::AnalysisShape------------------------------------------<<<"<<endl;
  cout<<endl;
  //*/
}



bool 
Model_Ctrl::
IsOneCoordDirrection(const gp_Dir& V, bool& isParallelX, bool& isParallelY, bool& isParallelZ)
{
  isParallelX = false;
  isParallelY = false;
  isParallelZ = false;

  bool result = false;

  gp_Dir xdir(1,0,0);
  gp_Dir ydir(0,1,0);
  gp_Dir zdir(0,0,1);

  if( V.IsOpposite(xdir,Precision::Angular())||V.IsParallel(xdir,Precision::Angular()) ){
    isParallelX = true;
  }else if( V.IsOpposite(ydir,Precision::Angular())||V.IsParallel(ydir,Precision::Angular()) ){
    isParallelY = true;
  }else if( V.IsOpposite(zdir,Precision::Angular())||V.IsParallel(zdir,Precision::Angular()) ){
    isParallelZ = true;
  }else{

  }
  result = isParallelX || isParallelY || isParallelZ;

  return result;
}


const map<Standard_Real, Standard_Integer, less<Standard_Real> >* 
Model_Ctrl::
GetSpecialCoordDatas(const Standard_Integer dir) const
{
  if(dir==0){
    return &m_SpecialXDatas;
  }else if(dir==1){
    return &m_SpecialYDatas;
  }else if(dir==2){
    return &m_SpecialZDatas;
  }else{
    cout<<"error-------------wrong dir in Model_Ctrl::GetSpecialCoordDatas --------------->>>"<<endl;
    return NULL;
  }
}



const vector<Standard_Real>* 
Model_Ctrl::
GetSpecialCoordVec(const Standard_Integer dir) const
{
  if(dir==0){
    return &m_SpecialXCoordVec;
  }else if(dir==1){
    return &m_SpecialYCoordVec;
  }else if(dir==2){
    return &m_SpecialZCoordVec;
  }else{
    cout<<"error-------------wrong dir in Model_Ctrl::GetSpecialCoordVec --------------->>>"<<endl;
    return NULL;
  }
}
