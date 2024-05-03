#ifndef _Mesh_Generation_HeaderFile
#define _Mesh_Generation_HeaderFile

#include <GeomBndDataDefine.hxx>
#include <OCCInclude.hxx>

#include <Grid_GenerationBase.hxx>
#include <GridBndData.hxx>

#include <TopTools_HSequenceOfShape.hxx>
#include <IntCurvesFace_NewShapeIntersector.hxx>
#include <IntCurvesFace_ShapeIntersector.hxx>
#include <IntCurvesFace_CFIntersector.hxx>
#include <TxSlab.h>



class Grid_Ctrl;
class Model_Ctrl;


#include <map>
using namespace std;

class Grid_Generation :public Grid_GenerationBase{

public:
  Grid_Generation();
  Grid_Generation(ZRGrid* _zrg, 
		  ZRDefine* _zrd, 
		  Model_Ctrl* _model_ctrl, 
		  const Standard_Integer _backgroundmaterialtype,
		  const Standard_Real _tol);
  ~Grid_Generation();


  void BuildGridBndDatas();

public:
  void SetModelCtrl(Model_Ctrl* _model_ctrl);
  void SetTolerance(const Standard_Real aParamt){ m_Tol = aParamt;};

public:
  void BuildModelsInformation();

//build EdgeBndPnt
public:
  void BuildEdgeBndPnts();
  void BuildEdgeBndPnts(const ZRGridLineDir aDir);
  void CheckEdgeBndPnts();
  void SetupEdgeBndPntDataMaterial(EdgeBndPntData& aData);


//build FaceBndPnt
public:
  void BuildFaceBndPnts();
  void CheckFaceBndPnts();
  void SetupFaceBndPntDataMaterial(FaceBndPntData& aData);


//convert EdgeBndPnt tp EdgeBndVertex
public:
  void EdgeBndPntConvertToEdgeBndVertex();
  void EdgeBndPntConvertToEdgeBndVertex(const ZRGridLineDir aDir);


//convert FaceBndPnt tp FaceBndVertex
public:
  void FaceBndPntConvertToFaceBndVertex();


// build Port and PeriodBnd common functions
public:
  void InsertEdgeBndVertexData(const ZRGridLineDir theRayDir, 
			       const Standard_Size theRayIndex, 
			       const EdgeBndVertexData& aBndVertex);

  void ExtendEdgeBndVerticesInRgn(const Standard_Integer thePortIndex,
				  const ZRGridLineDir& theDir, 
				  const Standard_Integer theRelativeDir, 
				  const TxSlab2D<Standard_Size>& theRgn);


public:
  void CleanFaceBndVerticesAcordingPort();
  void CleanFaceBndVerticesAcordingPort(const PortData& tmpPort);
  void CleanFaceBndVerticesInRgn(const Standard_Integer thePortIndex, const TxSlab2D<Standard_Size>& theRgn);
  
  // build Port functions
public:
  bool Is_EdgeBndPnt_OnOnePort(const EdgeBndPntData& theEdgeBndPnt);
  bool Is_FaceBndPnt_OnOnePort(const FaceBndPntData& theFaceBndPnt);

  void BuildPort();
  void ExtendBndsAccordingPorts();
  void ExtendBndsAccordingPort(const PortData& tmpPort);


//get methods
public:
  const map<Standard_Size,vector<EdgeBndPntData>,less<Standard_Size> > *    GetEdgeBndPntOf(const ZRGridLineDir aDir) const;
  map<Standard_Size,vector<EdgeBndPntData>,less<Standard_Size> > *    ModifyEdgeBndPntOf(const ZRGridLineDir aDir);

  const vector<FaceBndPntData>* GetFaceBndPnt() const;
  vector<FaceBndPntData>*    ModifyFaceBndPnt();

public:
  const Model_Ctrl* GetModelsCtrl() const {return m_ModelCtrl;};
  GridBndData* GetGridBndDatas() const {return m_GridBndDatas;};


private:
  Standard_Real m_Tol;
  Model_Ctrl* m_ModelCtrl;

  IntCurvesFace_ShapeIntersector m_ShapeIntersector;
  //IntCurvesFace_NewShapeIntersector m_ShapeIntersector;

  IntCurvesFace_CFIntersector m_CFIntersector;

  BRepClass_FaceClassifier m_Classifier;


public:
  GridBndData* m_GridBndDatas;

  map<Standard_Size,vector<EdgeBndPntData>,less<Standard_Size> > m_EdgeBndPnts0;
  map<Standard_Size,vector<EdgeBndPntData>,less<Standard_Size> > m_EdgeBndPnts1;

  vector<FaceBndPntData> m_FaceBndPnts;
};

#endif
