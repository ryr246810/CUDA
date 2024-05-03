#ifndef _ZRGrid_HeaderFile
#define _ZRGrid_HeaderFile

#include <TxVector.h>
#include <TxSlab2D.h>

#include <TxVector2D.h>
#include <TxVector.h>

#include <Standard_TypeDefine.hxx>
#include <map>

#include <UnitsSystemDef.hxx>

class ZRGrid
{
public:
  ZRGrid();
  ~ZRGrid();

  void Setup();

  void SetupGrid(const Standard_Real aPnt[2],
		 const vector<Standard_Real>& ZLengths,
		 const vector<Standard_Real>& RLengths,
		 const Standard_Integer theMargin,
		 const Standard_Integer thePMLLayer,
		 const Standard_Integer theResolution);


  void SetupGrid(const TxVector2D<Standard_Real>& aPnt,
		 const vector<Standard_Real>& ZLengths,
		 const vector<Standard_Real>& RLengths,
		 const Standard_Integer theMargin,
		 const Standard_Integer thePMLLayer,
		 const Standard_Integer theResolution);

  void ScaleAccordingUnitSystem(UnitsSystemDef* theUnitsSystem);

public:
  Standard_Size GetDimension(const Standard_Integer aDir) const;
  Standard_Size GetVertexDimension(const Standard_Integer aDir) const;
  Standard_Size GetEdgeDimension(const Standard_Integer edgeDir, const Standard_Integer dimDir) const;
  Standard_Size GetFaceDimension(const Standard_Integer dimDir) const;

  Standard_Size GetMaxIndexOfVertex(const Standard_Integer aDir) const;
  Standard_Size GetMaxIndexOfEdge(const Standard_Integer aDir, const Standard_Integer dimDir) const;
  Standard_Size GetMaxIndexOfFace(const Standard_Integer aDir) const;

  Standard_Size GetVertexSize() const;
  Standard_Size GetVertexSize(const Standard_Integer aDir ) const;

  Standard_Size GetFaceSize() const;
  Standard_Size GetFaceSize(const Standard_Integer aDir) const;

  Standard_Size GetEdgeSize(Standard_Integer aDir) const;
  Standard_Size GetEdgeSize(const Standard_Integer edgeDir, 
			    const Standard_Integer aDir) const;

public://。。。
  map<Standard_Integer, vector<Standard_Real>, less<Standard_Integer> >GetLVectors() const { return m_LVectors; };
  map<Standard_Integer, vector<Standard_Real>, less<Standard_Integer> >GetDLVectors() const { return m_DLVectors; };
  const Standard_Real * GetMinSteps() const { return m_MinSteps; };
  const Standard_Integer * GetDimensions() const { return m_Dimension; };
  //。。。
public:
  Standard_Integer GetPMLLayer() const {return m_PMLLayer;};
  Standard_Integer GetMargin() const {return m_Margin;};
  Standard_Integer GetResolutionRatio()const { return m_Resolution; }

  const TxSlab2D<Standard_Integer>& GetXtndRgn() const;
  const TxSlab2D<Standard_Integer>& GetPhysRgn() const;

public:
  TxVector2D<Standard_Real> GetOrg() const { return m_Org; };//。。。

  Standard_Real GetStep(const Standard_Integer aDir,
			const Standard_Size anIndex) const;
  
  Standard_Real GetDualStep(const Standard_Integer aDir, 
			    const Standard_Size indx) const;
  
  Standard_Real GetLength(const Standard_Integer aDir) const;
  
  Standard_Real GetLength(const Standard_Integer aDir,
			  const Standard_Size anIndex) const;
  
  Standard_Real GetCoordComp_From_VertexScalarIndx(const Standard_Integer dir, const Standard_Size indx) const;
  Standard_Real GetCoordComp_From_VertexVectorIndx(const Standard_Integer dir, const Standard_Size indxVec[2]) const;

  void GetCoord_From_VertexVectorIndx(const Standard_Size indxVec[2], Standard_Real coords[2]) const;
  void GetCoord_From_VertexScalarIndx(const Standard_Size indx, Standard_Real coords[2]) const;

  TxVector2D<Standard_Real> GetCoord_From_VertexVectorIndx(const Standard_Size _Index[2]) const;
  TxVector2D<Standard_Real> GetCoord_From_VertexScalarIndx(const Standard_Size indx) const;

  TxVector2D<Standard_Real> GetSteps(Standard_Size indx[2]) const;
  void GetSteps(Standard_Size indx[2], Standard_Real steps[2]) const;



public:
  void FillVertexIndxVec(const Standard_Size theVIndx,
			 Standard_Size theVIndxVec[2]) const;

  void FillVertexIndx(const Standard_Size theVIndxVec[2],
		      Standard_Size& theVIndx) const;
  
  void FillEdgeIndx(const Standard_Integer aDir,
		    const Standard_Size theIndxVec[2],
		    Standard_Size& theIndx) const;

  void FillEdgeIndxVec(const Standard_Integer aDir,
		       const Standard_Size theIndx,
		       Standard_Size theIndxVec[2]) const;
  
  void FillFaceIndx(const Standard_Size theVIndxVec[2],
		    Standard_Size& theVIndx) const;
  
  void FillFaceIndxVec(const Standard_Size theVIndx,
		       Standard_Size theVIndxVec[2]) const;  

public:
  void ComputeIndexVecAndWeightsInGrid(const TxVector2D<Standard_Real>& pos, 
				       TxVector2D<Standard_Size>& indx, 
				       TxVector2D<Standard_Real>& wl,
				       TxVector2D<Standard_Real>& wu) const;
  
  void ComputeIndexVecAndWeightsInGrid(Standard_Real pos[2], 
				       Standard_Size indx[2], 
				       Standard_Real wl[2],
				       Standard_Real wu[2]) const;

  void ComputeIndexVecAndWeightsInGrid(const TxVector2D<Standard_Real>& pos, 
				       Standard_Size indx[2], 
				       Standard_Real wl[2],
				       Standard_Real wu[2]) const;

public:
  void ComputeLocationOfEdgeBndPnt(const Standard_Integer aDir, 
				   const Standard_Real theZRPnt[2], 
				   bool& beInRgn, 
				   Standard_Size& theGridEdgeIndex, 
				   Standard_Size& theFrac) const;

  void ComputeLocationOfEdgeBndPnt(const Standard_Integer aDir, 
				   const TxVector2D<Standard_Real>& theZRPnt, 
				   bool& beInRgn, 
				   Standard_Size& theGridEdgeIndex, 
				   Standard_Size& theFrac) const;
  
  
  void ComputeLocationOfFaceBndPnt(const Standard_Real theZRPnt[2],
				   bool& beInRgn,
				   Standard_Size& theIndex1,
				   Standard_Size& theFrac1,
				   Standard_Size& theIndex2,
				   Standard_Size& theFrac2) const;

  void ComputeLocationOfFaceBndPnt(const TxVector2D<Standard_Real>& theZRPnt,
				   bool& beInRgn,
				   Standard_Size& theIndex1,
				   Standard_Size& theFrac1,
				   Standard_Size& theIndex2,
				   Standard_Size& theFrac2) const;


  void ComputeLocationInGrid(const TxVector2D<Standard_Real>& aLocation, 
			     Standard_Size theIndxVec[2]) const;

  void ComputeLocationInGrid(const TxVector2D<Standard_Real>& aLocation, 
			     TxVector2D<Standard_Size>& theIndxVec) const;

  void ComputeLocationInGrid(const Standard_Real aLocation[2], 
			     Standard_Size theIndxVec[2]) const;

  void ComputeLocationInGrid(const TxVector2D<Standard_Real>& aLocation, 
			     TxVector2D<Standard_Size>& theIndxVec, 
			     TxVector2D<Standard_Real>& thedLVec) const;


  void ComputeLocationInGrid(const TxVector2D<Standard_Real>& aLocation, 
			     TxVector2D<Standard_Size>& theIndxVec, 
			     TxVector2D<Standard_Size>& theFracVec) const;

  void ComputeLocationInGrid(const Standard_Real aLocation[2], 
			     Standard_Size theIndxVec[2],
			     Standard_Real thedLVec[2]) const;

  void ComputeLocationInGrid(const TxVector2D<Standard_Real>& aLocation, 
			     Standard_Size theIndxVec[2],
			     Standard_Real thedLVec[2]) const;


  void ComputeLocationInGrid(const Standard_Real aLocation[2], 
			     Standard_Size theIndxVec[2],
			     Standard_Size theFracVec[2]) const;

  void ComputeLocationInGrid(const TxVector2D<Standard_Real>& aLocation, 
			     Standard_Size theIndxVec[2],
			     Standard_Size theFracVec[2]) const;


  void ComputeLocationInGridInDir(const Standard_Integer& aDir, 
				  const Standard_Real& aLocation, 
				  Standard_Size& theIndx) const;

  void ComputeLocationInGridInDir(const Standard_Integer& aDir, 
				  const Standard_Real& aLocation, 
				  bool& beInRgn,
				  Standard_Size& theIndx,
				  Standard_Size& theFrac) const;

  void ComputeIndex(const Standard_Integer aDir, 
		    const Standard_Real aL, 
		    Standard_Size& theIndex ) const;

  void ComputeIndex2(const Standard_Integer aDir,
		     const Standard_Real aL,
		     Standard_Size& theIndex,
		     Standard_Real& thedL) const;

  void ComputeIndex2(const Standard_Integer aDir,
		     const Standard_Real aL, 
		     Standard_Size& theIndex,
		     Standard_Size& theFrac) const;

public:
  void ComputeBndBoxInGrid(const TxSlab2D<Standard_Real>& realRgn, 
			   TxSlab2D<Standard_Size>& gridRgn) const;

  void ExtendRgnToEndAlongDir(const Standard_Integer aDir,
			      const Standard_Integer aRelativeDir,
			      TxSlab2D<Standard_Size>& gridRgn) const;

  void ExtendRgnOneLayerAlongDir(const Standard_Integer aDir,
				 const Standard_Integer aRelativeDir,
				 TxSlab2D<Standard_Size>& gridRgn) const;

public:
  Standard_Size bumpFace(const Standard_Size dir,
			 const Standard_Size indx,
			 const Standard_Size amt) const;

  Standard_Size bumpFace(const Standard_Size dir,
			 const Standard_Size indx) const;
 
  Standard_Size bumpVertex(const Standard_Size dir,
			   const Standard_Size indx,
			   const Standard_Size amt) const;

  Standard_Size bumpVertex(const Standard_Size dir,
			   const Standard_Size indx) const;

  Standard_Size bumpEdge(const Standard_Size edgedir,
			 const Standard_Size bumpdir,
			 const Standard_Size indx,
			 const Standard_Size amt) const;

  Standard_Size bumpEdge(const Standard_Size edgedir,
			 const Standard_Size bumpdir,
			 const Standard_Size indx) const;



  Standard_Size iBumpFace(const Standard_Size dir,
			  const Standard_Size indx,
			  const Standard_Size amt) const;
  
  Standard_Size iBumpFace(const Standard_Size dir,
			  const Standard_Size indx) const;
 
  Standard_Size iBumpVertex(const Standard_Size dir,
			    const Standard_Size indx,
			    const Standard_Size amt) const;

  Standard_Size iBumpVertex(const Standard_Size dir,
			    const Standard_Size indx) const;

  Standard_Size iBumpEdge(const Standard_Size edgedir,
			  const Standard_Size bumpdir,
			  const Standard_Size indx,
			  const Standard_Size amt) const;

  Standard_Size iBumpEdge(const Standard_Size edgedir,
			  const Standard_Size bumpdir,
			  const Standard_Size indx) const;
  
  Standard_Size bumpVertexto(const Standard_Size theVertexIndex,
			     const Standard_Size bumpdir,
			     const Standard_Size bumpLocation) const;

  Standard_Size bumpEdgeto(const Standard_Size theEdgeIndex,
			   const Standard_Size edgedir,
			   const Standard_Size bumpdir,
			   const Standard_Size bumpLocation) const;

  Standard_Size bumpFaceto(const Standard_Size theFaceIndex,
			   const Standard_Size bumpdir,
			   const Standard_Size bumpLocation) const;

 
private:
  void SetOrg(const Standard_Real aPnt[2]);
  void SetOrg(const TxVector2D<Standard_Real>& _aPnt);
  void SetOrg(const Standard_Real _DirCoord_0, const Standard_Real _DirCoord_1);

  void SetMargin(const Standard_Integer _margin);
  void SetResolutionRatio( const Standard_Integer theResolution);
  void SetPMLLayer(const Standard_Integer _pmllayer) ;


  void SetupGridLengthsOf(const Standard_Integer aDir,
			  const vector<Standard_Real>& theLengths);

  void ComputeStep();
  void ComputeMinStep();
  void ComputeMinSteps();

  void ComputeSecondaryParams();
  void ComputeDimensions();

  void ComputeRealRgn();

  bool IsIn(const Standard_Integer aDir, const Standard_Real theLength) const;


public:
  Standard_Real GetMinStep() const;
  bool IsIn(const TxVector2D<Standard_Real>& thePnt) const;

  Standard_Real GetGridLengthEpsilon() const;

  const TxSlab2D<Standard_Real>& GetRealRgn() const;

private:
  TxVector2D<Standard_Real> m_Org;
  map<Standard_Integer, vector<Standard_Real>, less<Standard_Integer> > m_LVectors;

  Standard_Integer m_Margin;
  Standard_Integer m_Resolution;
  Standard_Integer m_PMLLayer;

private:
  map<Standard_Integer, vector<Standard_Real>, less<Standard_Integer> > m_DLVectors;

  Standard_Real m_Tol;
  Standard_Real m_MinStep;
  Standard_Real m_MinSteps[2];

  Standard_Integer   m_Dimension[2];
  Standard_Size m_VertexSizes[2];
  Standard_Size m_EdgeSizes[2][2];
  Standard_Size m_FaceSizes[2];

  Standard_Size m_XtndVSize;
  Standard_Size m_XtndFSize;
  Standard_Size m_XtndESize[2];

private:
  TxSlab2D<Standard_Integer> m_PhysRgn;
  TxSlab2D<Standard_Integer> m_XtndRgn;
  TxSlab2D<Standard_Real> m_RealRgn;

private:
 Standard_Size m_PhiNumber;


public:
  void SetPhiNumber(Standard_Size phi_Number) { m_PhiNumber = phi_Number;};

  Standard_Size GetPhiNumber() {return m_PhiNumber;};

  void ComputeIndexVecAndWeightsInGrid(TxVector<Standard_Real>& pos, 
				       TxVector<Standard_Size>& indx, 
				       TxVector<Standard_Real>& wl,
				       TxVector<Standard_Real>& wu) const;
  

  void ComputeIndexVecAndWeightsInGrid(TxVector<Standard_Real>& pos, 
				       Standard_Size indx[3], 
				       Standard_Real wl[3],
				       Standard_Real wu[3]) const;

  void ComputeLocationInGridPhi(TxVector<Standard_Real> & aLocation, 
			     Standard_Size& theIndxVec,
			     Standard_Real& theFracVec) const;

  void ComputeFactorCrossPhi(TxVector<Standard_Real> start_Loc,Standard_Size start_index,
		             TxVector<Standard_Real> end_Loc,Standard_Size end_index,
			        Standard_Real& mid_Loc)const;
};

#endif
