#ifndef _Mesh_Ctrl_Headerfile
#define _Mesh_Ctrl_Headerfile

#include <TxHierAttribSet.h>

#include <Model_Ctrl.hxx>
#include <ZRGrid.hxx>
#include <UnitsSystemDef.hxx>
#include <ZRDefine.hxx>


class Grid_Tool;


class ZRGrid_Ctrl
{
public:
  ZRGrid_Ctrl(UnitsSystemDef* theUnitsSystem, Model_Ctrl* theModelCtrl);
  ~ZRGrid_Ctrl();


public:
  void SetAttrib(const TxHierAttribSet& tha);
  void SetAttrib_Parameters(const TxHierAttribSet& tha);
  void SetAttrib_GridTools(const TxHierAttribSet& tha);

  void Build();
  void ClearGridTools();
  void CheckGridBuildTool();
  void CheckPorts();

  void CheckAndModifyDiscreteCoords();
  void CheckAndModifyDiscreteCoords(Standard_Integer dir);
  void CheckAndModifyDiscreteCoord(const Standard_Real refCoord, 
				   const Standard_Real currStep, 
				   Standard_Real& currCoord);


public:
  void ComputeBndBoxAccrodingInputShapes();
  void CheckMeshCtrl(bool& IsDefinedProperly);


public:
  void SetMinWaveLength(const Standard_Real _minwavelenght);
  void SetGeomResolutionRatio( const Standard_Integer theResolution); // for grid resolution ratio
  void SetExtendedNum(Standard_Integer theExtendedNum);


public:
  virtual void SetupGrid();
  virtual void SetupGrid(const Standard_Integer dir, Standard_Real& theOrg, vector<Standard_Real>& theNewLengths);


public:
  const Model_Ctrl* GetModelsCtrl() const;
  ZRGrid* GetZRGrid() const;
  ZRDefine* GetZRDefine() const;

  const TxSlab<Standard_Real>& GetBndBox() const;
  const Standard_Integer GetExtendedNum()const;

  const Standard_Integer GetGeomResolutionRatio() const;
  const Standard_Real GetMinWaveLength() const;

  const Standard_Integer GetMargin() const;
  void  SetMargin(const Standard_Integer _margin);


  Standard_Integer GetZDir() const;
  Standard_Integer GetRDir() const;
  Standard_Integer GetWorkPlaneDir() const;


private:
  void ExtendRgnDefineAccordingMargin(Standard_Integer& theLDExtendedStepNum, 
				      Standard_Integer& theRUExtendedStepNum);

  void ExtendRgnDefineAccordingPorts(const Standard_Integer dir,
				     Standard_Integer& theLDExtendedStepNum, 
				     Standard_Integer& theRUExtendedStepNum);

  void BuildNewLength(const Standard_Integer theLDExtendedStepNum,
		      const Standard_Real theLDStep,
		      const Standard_Integer theRUExtendedStepNum,
		      const Standard_Real theRUStep,
		      const vector<Standard_Real>& theMidLengths,
		      vector<Standard_Real>& theNewLengths);


protected:
  ZRGrid*                m_ZRGrid;
  ZRDefine*              m_ZRDefine;

private:
  UnitsSystemDef*            m_UnitsSystem;
  Model_Ctrl*                m_ModelsCtrl;

  TxSlab<Standard_Real>      m_BndBox;

  bool                       m_WaveLengthIsSet;
  Standard_Real              m_MinWaveLength;

  bool                       m_IsUniform;

  Standard_Integer           m_GeomResolutionRatio;

  Standard_Integer           m_ExtendedNum;
  Standard_Integer           m_Margin;

  // for port definition
  TxVector<bool> m_LowerBndsIsSetAsPort;
  TxVector<bool> m_UpperBndsIsSetAsPort;

  TxVector<bool> m_LowerBndsIsSetAsInputPort;
  TxVector<bool> m_UpperBndsIsSetAsInputPort;

  TxVector<Standard_Real> m_Org;

  Standard_Integer m_ZDir;
  Standard_Integer m_RDir;
  Standard_Integer m_WorkPlaneDir;

  TxVector<Standard_Real> m_ZUnitVec;
  TxVector<Standard_Real> m_RUnitVec;
  TxVector<Standard_Real> m_WorkPlaneUnitVec;

private:
  map<Standard_Integer, Grid_Tool*, less<Standard_Integer> > m_GridTools;

private:
  ZRGrid_Ctrl();
};

#endif
