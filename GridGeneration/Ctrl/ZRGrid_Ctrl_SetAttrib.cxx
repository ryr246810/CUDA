
#include <PhysConsts.hxx>
#include <PortDataFunc.hxx>

#include <ZRGrid_Ctrl.hxx>
#include <Grid_Tool.hxx>
#include <TxMakerMap.h>

void ZRGrid_Ctrl::
    SetAttrib(const TxHierAttribSet &tas)
{
    std::vector<std::string> gridCtrlNames = tas.getNamesOfType("GridCtrl");
    if (gridCtrlNames.empty())
    {
        cout << "error-----------------------------------No GridCtrl is defined" << endl;
        exit(1);
    }
    else
    {
        if (gridCtrlNames.size() > 1)
        {
            cout << "warning-----------------------------------GridCtrl are repeatedly defined" << endl;
        }
    }
    TxHierAttribSet tha = tas.getAttrib(gridCtrlNames[0]);

    SetAttrib_Parameters(tha);
    SetAttrib_GridTools(tha);

    CheckPorts();
}

void ZRGrid_Ctrl::
    SetAttrib_Parameters(const TxHierAttribSet &tha)
{
    if (tha.hasParam("frequency"))
    {
        Standard_Real frequency = tha.getParam("frequency");
        m_MinWaveLength = mksConsts.c / frequency;
        m_MinWaveLength = m_MinWaveLength / (m_UnitsSystem->GetRealUnitScaleOfLength());
    }
    else if (tha.hasParam("waveLength"))
    {
        m_MinWaveLength = tha.getParam("waveLength");
        m_MinWaveLength = m_MinWaveLength / (m_UnitsSystem->GetRealUnitScaleOfLength());
    }
    else
    {
        cout << "error--------------------------------no waveLength or frequency is definede" << endl;
    }

    if (tha.hasOption("geomResolutionRatio"))
    {
        m_GeomResolutionRatio = tha.getOption("geomResolutionRatio");
    }
    else
    {
        m_GeomResolutionRatio = 10;
    }

    if (tha.hasOption("margin"))
    {
        m_Margin = tha.getOption("margin");
    }
    else
    {
        m_Margin = 1;
    }

    if (tha.hasOption("extendedLayer"))
    {
        m_ExtendedNum = tha.getOption("extendedLayer");
    }
    else
    {
        m_ExtendedNum = 10;
    }

    if (tha.hasPrmVec("org"))
    {
        vector<Standard_Real> theOrgs = tha.getPrmVec("org");
        if (theOrgs.size() >= 3)
        {
            m_Org[0] = theOrgs[0];
            m_Org[1] = theOrgs[1];
            m_Org[2] = theOrgs[2];
        }
        else
        {
            for (Standard_Size i = 0; i < theOrgs.size(); i++)
            {
                m_Org[i] = theOrgs[i];
            }
        }
    }

    if (tha.hasOption("axis"))
    {
        m_ZDir = tha.getOption("axis");
        if (m_ZDir > 2 || m_ZDir < 0)
        {
            cout << "error----------------ZRGrid_Ctrl::SetAttrib_Parameters----------------1-1" << endl;
            exit(1);
        }
    }
    else
    {
        cout << "error----------------ZRGrid_Ctrl::SetAttrib_Parameters----------------1-2" << endl;
        exit(1);
    }

    if (tha.hasOption("rDir"))
    {
        m_RDir = tha.getOption("rDir");
        if (m_RDir > 2 || m_RDir < 0)
        {
            cout << "error----------------ZRGrid_Ctrl::SetAttrib_Parameters----------------2-1" << endl;
            exit(1);
        }
    }
    else
    {
        cout << "error----------------ZRGrid_Ctrl::SetAttrib_Parameters----------------2-2" << endl;
        exit(1);
    }
    m_WorkPlaneDir = 3 - m_ZDir - m_RDir;

    for (Standard_Integer dir = 0; dir < 3; dir++)
    {
        m_ZUnitVec[dir] = 0.0;
        m_RUnitVec[dir] = 0.0;
        m_WorkPlaneUnitVec[dir] = 0.0;
    }
    m_ZUnitVec[m_ZDir] = 1.0;
    m_RUnitVec[m_RDir] = 1.0;
    m_WorkPlaneUnitVec[m_WorkPlaneDir] = 1.0;

    ComputeBndBoxAccrodingInputShapes();
}

void ZRGrid_Ctrl::
    SetAttrib_GridTools(const TxHierAttribSet &tha)
{
    ClearGridTools();

    // Determine all the sources
    std::vector<std::string> gridDefineNames = tha.getNamesOfType("GridDefine");

    if (gridDefineNames.size())
    {
        std::cout << "\t GridDefines are:";
        for (size_t i = 0; i < gridDefineNames.size(); ++i)
            std::cout << " " << gridDefineNames[i];
        std::cout << std::endl;
    }

    for (size_t i = 0; i < gridDefineNames.size(); ++i)
    {
        TxHierAttribSet attribs = tha.getAttrib(gridDefineNames[i]);
        if (attribs.hasString("kind"))
        {
            std::string kind = attribs.getString("kind");
            Grid_Tool *oneNewTool = TxMakerMap<Grid_Tool>::getNew(kind);
            oneNewTool->SetGridCtrl(this);
            oneNewTool->SetAttrib(attribs);
            Standard_Integer dir = oneNewTool->GetDir();
            m_GridTools.insert(pair<Standard_Integer, Grid_Tool *>(dir, oneNewTool));
        }
    }

    CheckGridBuildTool();
}

/****************************************************************/
// Function :
// Purpose  :
/****************************************************************/
void ZRGrid_Ctrl::CheckPorts()
{
    m_LowerBndsIsSetAsPort = TxVector<bool>(false, false, false);
    m_UpperBndsIsSetAsPort = TxVector<bool>(false, false, false);

    gp_Pnt theBaryCenter;
    GridLineDir theLineDir;
    Standard_Integer theRelativeDir;

    const TColStd_DataMapOfIntegerInteger &thePorts = GetModelsCtrl()->GetPortsWithType();
    TColStd_DataMapIteratorOfDataMapOfIntegerInteger Iter;

    for (Iter.Initialize(thePorts); Iter.More(); Iter.Next())
    {
        Standard_Integer thePortIndex = Iter.Key();
        Standard_Integer thePortType = Iter.Value();

        if (!IsPMLPortType(thePortType))
            continue;

        GetModelsCtrl()->ComputePortDirWithFaceIndexOfPort(thePortIndex, theBaryCenter, theLineDir, theRelativeDir);

        if (theLineDir == DIRX)
        {
            if (theRelativeDir == 1)
            {
                if (!m_UpperBndsIsSetAsPort[0])
                    m_UpperBndsIsSetAsPort[0] = true;
            }
            else
            {
                if (!m_LowerBndsIsSetAsPort[0])
                    m_LowerBndsIsSetAsPort[0] = true;
            }
        }
        else if (theLineDir == DIRY)
        {
            if (theRelativeDir == 1)
            {
                if (!m_UpperBndsIsSetAsPort[1])
                    m_UpperBndsIsSetAsPort[1] = true;
            }
            else
            {
                if (!m_LowerBndsIsSetAsPort[1])
                    m_LowerBndsIsSetAsPort[1] = true;
            }
        }
        else if (theLineDir == DIRZ)
        {
            if (theRelativeDir == 1)
            {
                if (!m_UpperBndsIsSetAsPort[2])
                    m_UpperBndsIsSetAsPort[2] = true;
            }
            else
            {
                if (!m_LowerBndsIsSetAsPort[2])
                    m_LowerBndsIsSetAsPort[2] = true;
            }
        }
        else
        {
            cout << "ZRGrid_Ctrl::CheckNeededExtendedDir--------Error--------Port Dir is wrong" << endl;
        }

        m_UpperBndsIsSetAsInputPort = TxVector<bool>(false, false, false);
        m_LowerBndsIsSetAsInputPort = TxVector<bool>(false, false, false);

        if (IsInputPMLPortType(thePortType))
        {
            if (theLineDir == DIRX)
            {
                if (theRelativeDir == 1)
                {
                    if (!m_UpperBndsIsSetAsInputPort[0])
                        m_UpperBndsIsSetAsInputPort[0] = true;
                }
                else
                {
                    if (!m_LowerBndsIsSetAsInputPort[0])
                        m_LowerBndsIsSetAsInputPort[0] = true;
                }
            }
            else if (theLineDir == DIRY)
            {
                if (theRelativeDir == 1)
                {
                    if (!m_UpperBndsIsSetAsInputPort[1])
                        m_UpperBndsIsSetAsInputPort[1] = true;
                }
                else
                {
                    if (!m_LowerBndsIsSetAsInputPort[1])
                        m_LowerBndsIsSetAsInputPort[1] = true;
                }
            }
            else if (theLineDir == DIRZ)
            {
                if (theRelativeDir == 1)
                {
                    if (!m_UpperBndsIsSetAsInputPort[2])
                        m_UpperBndsIsSetAsInputPort[2] = true;
                }
                else
                {
                    if (!m_LowerBndsIsSetAsInputPort[2])
                        m_LowerBndsIsSetAsInputPort[2] = true;
                }
            }
            else
            {
                cout << "ZRGrid_Ctrl::CheckNeededExtendedDir--------Error--------InputPort Dir is wrong" << endl;
            }
        }
    }
}

void ZRGrid_Ctrl::
    CheckGridBuildTool()
{
    // this function will check:
    //        1. weather the tooles of all three direction are setted;
    //        2. weather the direction of the tooles are duplicated.
}
