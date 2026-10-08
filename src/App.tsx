import { Fragment, createContext, useContext, useMemo, useState } from "react";

/* =========================================================
   TYPES
========================================================= */

type Page =
  | "dashboard"
  | "incidents"
  | "evidence"
  | "timeline"
  | "conflicts"
  | "fusion"
  | "analysis";

type EvidenceSource =
  | "Temperature Sensor"
  | "Pressure Sensor"
  | "Vibration Sensor"
  | "System Logs"
  | "Maintenance Report"
  | "Operator Report";

type Severity = "low" | "medium" | "high" | "critical";

type IncidentStatus = "active" | "investigating" | "resolved";

type InvestigationTab =
  | "overview"
  | "evidence"
  | "timeline"
  | "conflicts"
  | "analysis";

type Evidence = {
  id: string;
  source: EvidenceSource;
  time: string;
  title: string;
  value: string;
  description: string;
  reliability: number;
  confidence: number;
  severity: Severity;
  mass?: MassMap;
};

type Incident = {
  id: string;
  equipment: string;
  title: string;
  status: IncidentStatus;
  severity: Severity;
  started: string;
  confidence: number;
  location: string;
  summary: string;
  rootCause: string;
  evidenceIds: string[];
};

type Conflict = {
  id: string;
  severity: Severity;
  status: "unresolved" | "resolved";
  sources: string;
  title: string;
  description: string;
  resolution: string;
};

/* =========================================================
   DEMPSTER-SHAFER TYPES
========================================================= */

type HypothesisKey =
  | "bearing"
  | "thermal"
  | "mechanical"
  | "lubrication"
  | "no_failure";

type MassMap = Record<string, number>;

type DSEvidenceResult = {
  evidenceId: string;
  reliability: number;
  rawMass: MassMap;
  discountedMass: MassMap;
};

type DSDiagnostic = {
  hypothesis: HypothesisKey;
  label: string;
  belief: number;
  plausibility: number;
  pignistic: number;
};

type DSResult = {
  fusedMass: MassMap;
  diagnostics: DSDiagnostic[];
  conflict: number;
  normalization: number;
  selectedCount: number;
  evidenceResults: DSEvidenceResult[];
};

const DS_HYPOTHESES: {
  key: HypothesisKey;
  label: string;
}[] = [
  {
    key: "bearing",
    label: "Bearing Degradation",
  },
  {
    key: "thermal",
    label: "Thermal Overload",
  },
  {
    key: "mechanical",
    label: "Mechanical Imbalance",
  },
  {
    key: "lubrication",
    label: "Lubrication Degradation",
  },
  {
    key: "no_failure",
    label: "No Confirmed Failure",
  },
];

const FRAME: HypothesisKey[] = DS_HYPOTHESES.map(
  (item) => item.key
);

const THETA = FRAME.join("|");

/* =========================================================
   DEMPSTER-SHAFER HELPERS
========================================================= */

function canonicalSet(values: string[]): string {
  return [...new Set(values)].sort().join("|");
}

function splitSet(key: string): string[] {
  if (!key) return [];
  return key.split("|");
}

function intersectSets(a: string, b: string): string {
  const setB = new Set(splitSet(b));

  return canonicalSet(
    splitSet(a).filter((value) => setB.has(value))
  );
}

function normalizeMassMap(mass: MassMap): MassMap {
  const result: MassMap = {};

  Object.entries(mass).forEach(([key, value]) => {
    if (value > 0.0000001) {
      result[key] = value;
    }
  });

  return result;
}

function discountMass(
  rawMass: MassMap,
  reliability: number
): MassMap {
  const r = Math.max(0, Math.min(1, reliability));

  const result: MassMap = {};

  Object.entries(rawMass).forEach(([set, mass]) => {
    if (set !== THETA) {
      result[set] = (result[set] ?? 0) + r * mass;
    }
  });

  const originalTheta = rawMass[THETA] ?? 0;

  result[THETA] =
    (result[THETA] ?? 0) +
    r * originalTheta +
    (1 - r);

  return normalizeMassMap(result);
}

function combineMasses(
  first: MassMap,
  second: MassMap
): {
  mass: MassMap;
  conflict: number;
} {
  const conjunctive: MassMap = {};
  let conflict = 0;

  Object.entries(first).forEach(([setA, massA]) => {
    Object.entries(second).forEach(([setB, massB]) => {
      const product = massA * massB;

      const intersection = intersectSets(setA, setB);

      if (!intersection) {
        conflict += product;
      } else {
        conjunctive[intersection] =
          (conjunctive[intersection] ?? 0) + product;
      }
    });
  });

  const denominator = 1 - conflict;

  if (denominator <= 0.0000001) {
    return {
      mass: {
        [THETA]: 1,
      },
      conflict: 1,
    };
  }

  const normalized: MassMap = {};

  Object.entries(conjunctive).forEach(([set, mass]) => {
    normalized[set] = mass / denominator;
  });

  return {
    mass: normalizeMassMap(normalized),
    conflict,
  };
}

function calculateBelief(
  mass: MassMap,
  hypothesis: HypothesisKey
): number {
  let belief = 0;

  Object.entries(mass).forEach(([set, value]) => {
    const members = splitSet(set);

    if (
      members.length > 0 &&
      members.every((member) => member === hypothesis)
    ) {
      belief += value;
    }
  });

  return belief;
}

function calculatePlausibility(
  mass: MassMap,
  hypothesis: HypothesisKey
): number {
  let plausibility = 0;

  Object.entries(mass).forEach(([set, value]) => {
    const members = splitSet(set);

    if (members.includes(hypothesis)) {
      plausibility += value;
    }
  });

  return plausibility;
}

function calculatePignistic(
  mass: MassMap,
  hypothesis: HypothesisKey
): number {
  let probability = 0;

  Object.entries(mass).forEach(([set, value]) => {
    const members = splitSet(set);

    if (
      members.length > 0 &&
      members.includes(hypothesis)
    ) {
      probability += value / members.length;
    }
  });

  return probability;
}

/* =========================================================
   EXPLICIT MASS ASSIGNMENTS
========================================================= */

const MASS_ASSIGNMENTS: Record<string, MassMap> = {
  "E-104": {
    thermal: 0.45,
    bearing: 0.25,
    "bearing|thermal": 0.15,
    [THETA]: 0.15,
  },

  "E-105": {
    thermal: 0.2,
    bearing: 0.15,
    mechanical: 0.1,
    [THETA]: 0.55,
  },

  "E-106": {
    bearing: 0.4,
    mechanical: 0.3,
    "bearing|mechanical": 0.15,
    [THETA]: 0.15,
  },

  "E-107": {
    bearing: 0.7,
    "bearing|thermal": 0.05,
    [THETA]: 0.25,
  },

  "E-108": {
    bearing: 0.55,
    lubrication: 0.2,
    "bearing|lubrication": 0.1,
    [THETA]: 0.15,
  },

  "E-109": {
    no_failure: 0.4,
    [THETA]: 0.6,
  },

  "E-110": {
    thermal: 0.55,
    bearing: 0.25,
    "bearing|thermal": 0.1,
    [THETA]: 0.1,
  },
};

/* =========================================================
   RUN DEMPSTER-SHAFER FUSION
========================================================= */

function runDempsterShaferFusion(
  items: Evidence[],
  selectedIds: string[],
  reliabilityWeights: Record<string, number>
): DSResult {
  const selectedItems = items.filter((item) =>
    selectedIds.includes(item.id)
  );

  if (selectedItems.length === 0) {
    return {
      fusedMass: {
        [THETA]: 1,
      },
      diagnostics: DS_HYPOTHESES.map((hypothesis) => ({
        hypothesis: hypothesis.key,
        label: hypothesis.label,
        belief: 0,
        plausibility: 1,
        pignistic: 1 / FRAME.length,
      })),
      conflict: 0,
      normalization: 1,
      selectedCount: 0,
      evidenceResults: [],
    };
  }

  const evidenceResults: DSEvidenceResult[] =
    selectedItems.map((item) => {
      const rawMass =
        item.mass ?? MASS_ASSIGNMENTS[item.id] ?? {
          [THETA]: 1,
        };

      const reliability =
        (reliabilityWeights[item.id] ??
          item.reliability) / 100;

      return {
        evidenceId: item.id,
        reliability: reliability * 100,
        rawMass,
        discountedMass: discountMass(
          rawMass,
          reliability
        ),
      };
    });

  let fusedMass = evidenceResults[0].discountedMass;
  let totalConflict = 0;

  for (let i = 1; i < evidenceResults.length; i++) {
    const combination = combineMasses(
      fusedMass,
      evidenceResults[i].discountedMass
    );

    totalConflict =
      1 -
      (1 - totalConflict) *
        (1 - combination.conflict);

    fusedMass = combination.mass;
  }

  const diagnostics = DS_HYPOTHESES.map(
    (hypothesis) => ({
      hypothesis: hypothesis.key,
      label: hypothesis.label,
      belief: calculateBelief(
        fusedMass,
        hypothesis.key
      ),
      plausibility: calculatePlausibility(
        fusedMass,
        hypothesis.key
      ),
      pignistic: calculatePignistic(
        fusedMass,
        hypothesis.key
      ),
    })
  ).sort(
    (a, b) => b.pignistic - a.pignistic
  );

  return {
    fusedMass,
    diagnostics,
    conflict: totalConflict,
    normalization: 1 - totalConflict,
    selectedCount: selectedItems.length,
    evidenceResults,
  };
}

/* =========================================================
   MOCK DATA
========================================================= */

const baseEvidenceData: Evidence[] = [
  {
    id: "E-104",
    source: "Temperature Sensor",
    time: "14:34:21",
    title: "Temperature spike",
    value: "118.4 °C",
    description:
      "Bearing housing temperature increased rapidly above the expected operating range.",
    reliability: 94,
    confidence: 89,
    severity: "high",
  },
  {
    id: "E-105",
    source: "Pressure Sensor",
    time: "14:31:08",
    title: "Pressure fluctuation",
    value: "8.7 bar",
    description:
      "Short-duration pressure fluctuation detected during turbine operation.",
    reliability: 91,
    confidence: 84,
    severity: "medium",
  },
  {
    id: "E-106",
    source: "Vibration Sensor",
    time: "14:27:42",
    title: "Vibration increase",
    value: "4.2× baseline",
    description:
      "Vibration amplitude increased substantially relative to the normal baseline.",
    reliability: 93,
    confidence: 92,
    severity: "high",
  },
  {
    id: "E-107",
    source: "System Logs",
    time: "14:21:03",
    title: "Bearing warning",
    value: "WARN-204",
    description:
      "Control system generated a bearing condition warning before the main anomaly.",
    reliability: 88,
    confidence: 81,
    severity: "medium",
  },
  {
    id: "E-108",
    source: "Maintenance Report",
    time: "13:45:00",
    title: "Previous bearing wear",
    value: "Moderate",
    description:
      "Maintenance documentation recorded moderate wear on the turbine bearing.",
    reliability: 86,
    confidence: 78,
    severity: "medium",
  },
  {
    id: "E-109",
    source: "Operator Report",
    time: "14:36:10",
    title: "Normal operating sound",
    value: "No anomaly",
    description:
      "Operator reported no abnormal sound immediately before the alarm.",
    reliability: 62,
    confidence: 54,
    severity: "low",
  },
  {
    id: "E-110",
    source: "System Logs",
    time: "14:37:02",
    title: "High-temperature alarm",
    value: "ALARM-701",
    description:
      "Emergency temperature threshold was exceeded and an alarm was triggered.",
    reliability: 97,
    confidence: 96,
    severity: "critical",
  },
];

const baseIncidents: Incident[] = [
  {
    id: "INC-2026-014",
    equipment: "Turbine T-04",
    title: "Bearing failure",
    status: "active",
    severity: "critical",
    started: "14:21:03",
    confidence: 87.4,
    location: "Generation Unit A",
    summary:
      "Multiple heterogeneous evidence sources indicate progressive bearing degradation followed by vibration and thermal escalation.",
    rootCause: "Progressive bearing degradation",
    evidenceIds: [
      "E-104",
      "E-105",
      "E-106",
      "E-107",
      "E-108",
      "E-109",
      "E-110",
    ],
  },
  {
    id: "INC-2026-013",
    equipment: "Pump P-02",
    title: "Discharge pressure anomaly",
    status: "investigating",
    severity: "high",
    started: "11:08:42",
    confidence: 72.8,
    location: "Process Hall B",
    summary:
      "Pressure fluctuation is being investigated alongside system warning evidence.",
    rootCause: "Undetermined",
    evidenceIds: ["E-105", "E-107"],
  },
  {
    id: "INC-2026-012",
    equipment: "Compressor C-07",
    title: "Excessive vibration",
    status: "resolved",
    severity: "medium",
    started: "09:42:17",
    confidence: 91.2,
    location: "Compression Unit",
    summary:
      "Vibration evidence was correlated with a previously documented mechanical maintenance condition.",
    rootCause: "Mechanical imbalance",
    evidenceIds: ["E-106", "E-108"],
  },
];

const baseConflicts: Conflict[] = [
  {
    id: "C-001",
    severity: "high",
    status: "resolved",
    sources: "E-106 ↔ E-109",
    title: "Sensor vs operator observation",
    description:
      "The vibration sensor recorded a 4.2× increase while the operator reported normal operating sound.",
    resolution:
      "Sensor evidence receives greater influence because of its higher reliability and quantitative measurement.",
  },
  {
    id: "C-002",
    severity: "medium",
    status: "resolved",
    sources: "E-105 ↔ E-104",
    title: "Pressure-temperature timing",
    description:
      "Pressure fluctuation occurred before the major temperature escalation.",
    resolution:
      "Pressure is treated as a contributing precursor rather than the primary failure mechanism.",
  },
  {
    id: "C-003",
    severity: "medium",
    status: "unresolved",
    sources: "E-107 ↔ E-108",
    title: "Warning vs maintenance history",
    description:
      "The system warning and previous maintenance observation support bearing degradation, but their exact causal relationship remains uncertain.",
    resolution:
      "Requires additional maintenance history or inspection evidence.",
  },
];


/* =========================================================
   MULTI-CASE RESEARCH DATA MODEL
========================================================= */

type ResearchCase = {
  id: string;
  name: string;
  equipment: string;
  studyType: string;
  description: string;
  evidence: Evidence[];
  incidents: Incident[];
  conflicts: Conflict[];
};

const caseStudies: ResearchCase[] = [
  {
    id: "CASE-001",
    name: "Turbine Bearing Failure",
    equipment: "Turbine T-04",
    studyType: "Industrial incident",
    description:
      "Progressive bearing degradation with vibration and thermal escalation.",
    evidence: baseEvidenceData,
    incidents: baseIncidents,
    conflicts: baseConflicts,
  },
  {
    id: "CASE-002",
    name: "Compressor Vibration Study",
    equipment: "Compressor C-12",
    studyType: "Research case study",
    description:
      "A separate vibration-focused case used to demonstrate that the same fusion engine can analyze a different evidence set.",
    evidence: [
      {
        id: "E-201",
        source: "Vibration Sensor",
        time: "10:14:12",
        title: "Rotor vibration increase",
        value: "6.8× baseline",
        description: "Sustained vibration increase detected at the compressor rotor.",
        reliability: 95,
        confidence: 93,
        severity: "critical",
        mass: { mechanical: 0.62, bearing: 0.12, "bearing|mechanical": 0.11, [THETA]: 0.15 },
      },
      {
        id: "E-202",
        source: "Temperature Sensor",
        time: "10:15:01",
        title: "Bearing temperature rise",
        value: "96.2 °C",
        description: "Bearing temperature rose above the normal operating envelope.",
        reliability: 92,
        confidence: 86,
        severity: "high",
        mass: { thermal: 0.35, bearing: 0.25, mechanical: 0.15, [THETA]: 0.25 },
      },
      {
        id: "E-203",
        source: "System Logs",
        time: "10:15:44",
        title: "Rotor imbalance warning",
        value: "VIB-302",
        description: "Control software reported a rotor imbalance condition.",
        reliability: 94,
        confidence: 91,
        severity: "high",
        mass: { mechanical: 0.72, "bearing|mechanical": 0.08, [THETA]: 0.20 },
      },
      {
        id: "E-204",
        source: "Maintenance Report",
        time: "09:20:00",
        title: "Coupling wear",
        value: "Moderate",
        description: "Previous inspection recorded moderate coupling wear.",
        reliability: 84,
        confidence: 77,
        severity: "medium",
        mass: { mechanical: 0.50, lubrication: 0.10, "bearing|mechanical": 0.10, [THETA]: 0.30 },
      },
      {
        id: "E-205",
        source: "Operator Report",
        time: "10:16:20",
        title: "Strong vibration felt",
        value: "Confirmed",
        description: "Operator reported noticeable vibration during the event.",
        reliability: 72,
        confidence: 75,
        severity: "medium",
        mass: { mechanical: 0.50, bearing: 0.10, [THETA]: 0.40 },
      },
    ],
    incidents: [
      {
        id: "INC-CASE2-001",
        equipment: "Compressor C-12",
        title: "Excessive rotor vibration",
        status: "investigating",
        severity: "high",
        started: "10:14:12",
        confidence: 0,
        location: "Compression Unit B",
        summary: "A new case study with independent evidence is being fused to determine the dominant failure hypothesis.",
        rootCause: "Pending D-S fusion",
        evidenceIds: ["E-201", "E-202", "E-203", "E-204", "E-205"],
      },
    ],
    conflicts: [
      {
        id: "C-201",
        severity: "low",
        status: "resolved",
        sources: "E-201 ↔ E-202",
        title: "Vibration vs thermal evidence",
        description: "Both sources indicate abnormal operation, but their strongest hypotheses differ.",
        resolution: "Fusion retains both signals and lets their reliability-weighted masses determine the result.",
      },
    ],
  },
  {
    id: "CASE-003",
    name: "Pump Pressure Anomaly",
    equipment: "Pump P-07",
    studyType: "Research case study",
    description:
      "Pressure instability with supporting thermal and operator evidence.",
    evidence: [
      {
        id: "E-301",
        source: "Pressure Sensor",
        time: "16:02:10",
        title: "Discharge pressure drop",
        value: "4.1 bar",
        description: "Discharge pressure dropped below the expected operating range.",
        reliability: 96,
        confidence: 94,
        severity: "critical",
        mass: { mechanical: 0.25, thermal: 0.08, [THETA]: 0.67 },
      },
      {
        id: "E-302",
        source: "Temperature Sensor",
        time: "16:03:05",
        title: "Pump temperature rise",
        value: "88.6 °C",
        description: "Pump casing temperature increased during the pressure anomaly.",
        reliability: 90,
        confidence: 83,
        severity: "medium",
        mass: { thermal: 0.48, mechanical: 0.17, [THETA]: 0.35 },
      },
      {
        id: "E-303",
        source: "System Logs",
        time: "16:03:44",
        title: "Low-flow warning",
        value: "FLOW-118",
        description: "Control system reported a low-flow condition.",
        reliability: 95,
        confidence: 92,
        severity: "high",
        mass: { mechanical: 0.35, lubrication: 0.10, thermal: 0.10, [THETA]: 0.45 },
      },
      {
        id: "E-304",
        source: "Operator Report",
        time: "16:04:01",
        title: "Flow irregularity observed",
        value: "Confirmed",
        description: "Operator reported intermittent flow behavior.",
        reliability: 76,
        confidence: 79,
        severity: "medium",
        mass: { mechanical: 0.35, thermal: 0.10, [THETA]: 0.55 },
      },
    ],
    incidents: [
      {
        id: "INC-CASE3-001",
        equipment: "Pump P-07",
        title: "Discharge pressure anomaly",
        status: "active",
        severity: "high",
        started: "16:02:10",
        confidence: 0,
        location: "Process Hall C",
        summary: "Independent pump evidence is processed to determine the most supported explanation for the pressure event.",
        rootCause: "Pending D-S fusion",
        evidenceIds: ["E-301", "E-302", "E-303", "E-304"],
      },
    ],
    conflicts: [],
  },
];


function enrichResearchCase(sourceCase: ResearchCase): ResearchCase {
  const incidents = sourceCase.incidents.map((incident) => {
    const items = sourceCase.evidence.filter((item) =>
      incident.evidenceIds.includes(item.id)
    );
    const weights = Object.fromEntries(
      items.map((item) => [item.id, item.reliability])
    );
    const result = runDempsterShaferFusion(
      items,
      items.map((item) => item.id),
      weights
    );
    const primary = result.diagnostics[0];

    return {
      ...incident,
      confidence: Number(((primary?.pignistic ?? 0) * 100).toFixed(1)),
      rootCause: primary?.label ?? incident.rootCause,
    };
  });

  return {
    ...sourceCase,
    incidents,
  };
}

const ResearchCaseContext = createContext<ResearchCase | null>(null);

function useResearchCase(): ResearchCase {
  const value = useContext(ResearchCaseContext);
  if (!value) {
    throw new Error("useResearchCase must be used inside ResearchCaseContext.Provider");
  }
  return value;
}

/* =========================================================
   MAIN APP
========================================================= */

function App() {
  const [activePage, setActivePage] =
    useState<Page>("dashboard");

  const [selectedCaseId, setSelectedCaseId] =
    useState("CASE-001");

  const rawCase =
    caseStudies.find((item) => item.id === selectedCaseId) ?? caseStudies[0];

  const activeCase = useMemo(
    () => enrichResearchCase(rawCase),
    [rawCase]
  );

  const [selectedIncidentId, setSelectedIncidentId] =
    useState(activeCase.incidents[0]?.id ?? "");

  const [investigationTab, setInvestigationTab] =
    useState<InvestigationTab>("overview");

  const [selectedEvidenceId, setSelectedEvidenceId] =
    useState("E-110");

  const selectedIncident =
    activeCase.incidents.find(
      (incident) =>
        incident.id === selectedIncidentId
    ) ?? activeCase.incidents[0];

  const openIncident = (incidentId: string) => {
    setSelectedIncidentId(incidentId);
    const incident = activeCase.incidents.find((item) => item.id === incidentId);
    setSelectedEvidenceId(incident?.evidenceIds[0] ?? activeCase.evidence[0]?.id ?? "");
    setInvestigationTab("overview");
    setActivePage("incidents");
  };

  const openInvestigationTab = (
    tab: InvestigationTab,
    incidentId = selectedIncidentId
  ) => {
    setSelectedIncidentId(incidentId);
    setInvestigationTab(tab);
    setActivePage("incidents");
  };

  return (
    <div className="app-shell">
      <Sidebar
        activePage={activePage}
        onNavigate={(page) => {
          if (page === "incidents") {
            const firstIncident = activeCase.incidents[0];
            if (firstIncident) {
              setSelectedIncidentId(firstIncident.id);
              setSelectedEvidenceId(firstIncident.evidenceIds[0] ?? activeCase.evidence[0]?.id ?? "");
            }
            setInvestigationTab("overview");
          }
          setActivePage(page);
        }}
      />

      <ResearchCaseContext.Provider value={activeCase}>
      <main className="main-content">
        <Topbar
          activeCase={activeCase}
          cases={caseStudies}
          onCaseChange={(caseId) => {
            setSelectedCaseId(caseId);
            const nextCase = caseStudies.find((item) => item.id === caseId) ?? caseStudies[0];
            setSelectedIncidentId(nextCase.incidents[0]?.id ?? "");
            setSelectedEvidenceId(nextCase.evidence[0]?.id ?? "");
          }}
        />

        {activePage === "dashboard" && (
          <Dashboard
            incident={selectedIncident}
            onOpenIncident={() =>
              openIncident(selectedIncident.id)
            }
            onOpenIncidents={() =>
              setActivePage("incidents")
            }
          />
        )}

        {activePage === "incidents" && (
          <IncidentsPage
            selectedIncident={selectedIncident}
            investigationTab={investigationTab}
            onSelectIncident={openIncident}
            onBack={() =>
              setInvestigationTab("overview")
            }
            onTabChange={openInvestigationTab}
            selectedEvidenceId={selectedEvidenceId}
            setSelectedEvidenceId={
              setSelectedEvidenceId
            }
          />
        )}

        {activePage === "evidence" && (
          <EvidenceExplorer
            selectedEvidenceId={selectedEvidenceId}
            setSelectedEvidenceId={
              setSelectedEvidenceId
            }
          />
        )}

        {activePage === "timeline" && (
          <TimelinePage
            incident={selectedIncident}
            onOpenInvestigation={() =>
              openInvestigationTab(
                "timeline",
                selectedIncident.id
              )
            }
          />
        )}

        {activePage === "conflicts" && (
          <ConflictsPage
            incident={selectedIncident}
            onOpenInvestigation={() =>
              openInvestigationTab(                "conflicts",
                selectedIncident.id
              )
            }
          />
        )}

        {activePage === "fusion" && (
          <FusionEnginePage
            key={selectedIncident.id}
            incident={selectedIncident}
          />
        )}

        {activePage === "analysis" && (
          <FailureAnalysisPage
            key={selectedIncident.id}
            incident={selectedIncident}
          />
        )}
      </main>
      </ResearchCaseContext.Provider>
    </div>
  );
}

/* =========================================================
   SIDEBAR
========================================================= */

function Sidebar({
  activePage,
  onNavigate,
}: {
  activePage: Page;
  onNavigate: (page: Page) => void;
}) {
  const groups = [
    {
      label: "OVERVIEW",
      items: [
        {
          id: "dashboard" as Page,
          label: "Dashboard",
          icon: "⌂",
        },
        {
          id: "incidents" as Page,
          label: "Incidents",
          icon: "!",
        },
      ],
    },
    {
      label: "EVIDENCE",
      items: [
        {
          id: "evidence" as Page,
          label: "Evidence Explorer",
          icon: "◈",
        },
        {
          id: "timeline" as Page,
          label: "Timeline",
          icon: "◷",
        },
        {
          id: "conflicts" as Page,
          label: "Conflicts",
          icon: "⚠",
        },
      ],
    },
    {
      label: "ANALYSIS",
      items: [
        {
          id: "fusion" as Page,
          label: "Fusion Engine",
          icon: "◎",
        },
        {
          id: "analysis" as Page,
          label: "Failure Analysis",
          icon: "⌁",
        },
      ],
    },
  ];

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">EF</div>

        <div>
          <div className="brand-title">
            Evidence Fusion
          </div>

          <div className="brand-subtitle">
            Incident Intelligence
          </div>
        </div>
      </div>

      <div className="sidebar-divider" />

      <nav>
        {groups.map((group) => (
          <div
            className="nav-group"
            key={group.label}
          >
            <div className="nav-group-title">
              {group.label}
            </div>

            {group.items.map((item) => (
              <button
                key={item.id}
                className={`nav-item ${
                  activePage === item.id
                    ? "active"
                    : ""
                }`}
                onClick={() =>
                  onNavigate(item.id)
                }
              >
                <span className="nav-icon">
                  {item.icon}
                </span>

                <span>{item.label}</span>
              </button>
            ))}
          </div>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="system-indicator">
          <span className="online-dot" />
          <span>Fusion Engine Online</span>
        </div>

        <div className="version">
          v2.0.0 • Dempster-Shafer Prototype
        </div>
      </div>
    </aside>
  );
}

/* =========================================================
   TOPBAR
========================================================= */

function Topbar({
  activeCase,
  cases,
  onCaseChange,
}: {
  activeCase: ResearchCase;
  cases: ResearchCase[];
  onCaseChange: (caseId: string) => void;
}) {
  return (
    <header className="topbar">
      <div>
        <div className="topbar-title">
          Industrial Evidence Intelligence
        </div>

        <div className="topbar-subtitle">
          Multi-source incident reconstruction and
          failure analysis
        </div>
      </div>

      <div className="topbar-case-selector">
        <span className="eyebrow">ACTIVE CASE</span>
        <select
          value={activeCase.id}
          onChange={(event) => onCaseChange(event.target.value)}
        >
          {cases.map((item) => (
            <option key={item.id} value={item.id}>
              {item.id} • {item.name}
            </option>
          ))}
        </select>
      </div>

      <div className="topbar-status">
        <span className="online-dot" />
        SYSTEM ONLINE
      </div>
    </header>
  );
}

/* =========================================================
   DASHBOARD
========================================================= */

function Dashboard({
  incident,
  onOpenIncident,
  onOpenIncidents,
}: {
  incident: Incident;
  onOpenIncident: () => void;
  onOpenIncidents: () => void;
}) {
  const { evidence: evidenceData, conflicts } = useResearchCase();
  const incidentEvidence = evidenceData.filter((item) =>
    incident?.evidenceIds.includes(item.id)
  );

  const weights = Object.fromEntries(
    incidentEvidence.map((item) => [item.id, item.reliability])
  );

  const fusion = runDempsterShaferFusion(
    incidentEvidence,
    incidentEvidence.map((item) => item.id),
    weights
  );

  const primary = fusion.diagnostics[0];
  const primaryPercent = (primary?.pignistic ?? 0) * 100;
  const beliefPercent = (primary?.belief ?? 0) * 100;
  const plausibilityPercent = (primary?.plausibility ?? 0) * 100;
  const conflictPercent = fusion.conflict * 100;

  const sensorEvidence = incidentEvidence.slice(0, 4).map((item, index) => ({
    id: item.id,
    label: item.source.replace(" Sensor", ""),
    value: item.value,
    reliability: item.reliability,
    status: item.severity.toUpperCase(),
    position: ["sensor-top", "sensor-left", "sensor-bottom", "sensor-right"][index] ?? "sensor-top",
  }));

  const sequence = [...incidentEvidence].sort((a, b) =>
    a.time.localeCompare(b.time)
  );

  return (
    <div className="command-center">
      <div className="command-header">
        <div>
          <div className="eyebrow">INDUSTRIAL COMMAND CENTER</div>
          <h1>Incident Overview</h1>
          <p>
            Case-aware reconstruction of equipment condition from heterogeneous evidence.
          </p>
        </div>

        <div className="command-actions">
          <div className="live-indicator">
            <span className="live-pulse" />
            CASE ACTIVE
          </div>
          <button className="command-button" onClick={onOpenIncident}>
            Open Investigation →
          </button>
        </div>
      </div>

      <section className="industrial-hero">
        <div className="industrial-grid" />

        <div className="hero-topline">
          <div>
            <span className="incident-code">{incident?.id ?? "NO-INCIDENT"}</span>
            <span className="hero-divider">/</span>
            <span>{incident?.equipment ?? "NO EQUIPMENT"}</span>
          </div>

          <div className="critical-status">
            <span />
            {(incident?.severity ?? "low").toUpperCase()} INCIDENT
          </div>
        </div>

        {sensorEvidence.map((sensor) => (
          <div className={`sensor-node ${sensor.position}`} key={sensor.id}>
            <div className="sensor-node-header">
              <span className="sensor-node-id">{sensor.id}</span>
              <span className="sensor-node-status">{sensor.status}</span>
            </div>
            <div className="sensor-node-value">
              {sensor.value}
            </div>
            <div className="sensor-node-footer">
              Reliability {sensor.reliability}%
            </div>
          </div>
        ))}

        <div className="signal-line signal-one" />
        <div className="signal-line signal-two" />
        <div className="signal-line signal-three" />
        <div className="signal-line signal-four" />

        <div className="turbine-stage">
          <div className="turbine-glow" />
          <div className="turbine">
            <div className="turbine-ring ring-one" />
            <div className="turbine-ring ring-two" />
            <div className="turbine-ring ring-three" />
            <div className="turbine-body">
              <div className="turbine-body-top" />
              <div className="turbine-body-core">
                <div className="core-light" />
                <span>EF</span>
              </div>
              <div className="turbine-body-bottom" />
            </div>
            <div className="turbine-blade blade-one" />
            <div className="turbine-blade blade-two" />
            <div className="turbine-blade blade-three" />
            <div className="turbine-blade blade-four" />
            <div className="turbine-hub" />
          </div>
          <div className="equipment-label">
            <strong>{incident?.equipment ?? "UNKNOWN EQUIPMENT"}</strong>
            <span>{incident?.location ?? "UNKNOWN LOCATION"}</span>
          </div>
        </div>

        <div className="hero-incident-title">
          <div className="hero-title-kicker">FAILURE RECONSTRUCTION</div>
          <h2>{primary?.label ?? incident?.rootCause ?? "Awaiting evidence"}</h2>
          <p>{incident?.summary ?? "Add evidence to begin incident reconstruction."}</p>
        </div>

        <div className="hero-fusion">
          <div className="hero-fusion-label">CURRENT FUSION</div>
          <div className="hero-fusion-value">
            {primaryPercent.toFixed(1)}<span>%</span>
          </div>
          <div className="hero-fusion-sub">
            {primary?.label ?? "No fused hypothesis"}
          </div>
          <div className="hero-fusion-bar">
            <span style={{ width: `${Math.min(100, primaryPercent)}%` }} />
          </div>
          <div className="hero-fusion-meta">
            <span><strong>{beliefPercent.toFixed(1)}%</strong>Belief</span>
            <span><strong>{plausibilityPercent.toFixed(1)}%</strong>Plausibility</span>
            <span><strong>{conflictPercent.toFixed(1)}%</strong>Conflict</span>
          </div>
        </div>
      </section>

      <div className="incident-status-strip">
        <div className="status-block">
          <span>INCIDENT STATUS</span>
          <strong className="status-critical">{incident?.status?.toUpperCase() ?? "N/A"}</strong>
        </div>
        <div className="status-block">
          <span>EQUIPMENT</span>
          <strong>{incident?.equipment ?? "—"}</strong>
        </div>
        <div className="status-block">
          <span>LOCATION</span>
          <strong>{incident?.location ?? "—"}</strong>
        </div>
        <div className="status-block">
          <span>STARTED</span>
          <strong>{incident?.started ?? "—"}</strong>
        </div>
        <div className="status-block">
          <span>EVIDENCE SOURCES</span>
          <strong>{incidentEvidence.length} / {incidentEvidence.length}</strong>
        </div>
      </div>

      <div className="command-grid">
        <section className="command-panel signal-panel">
          <div className="command-panel-header">
            <div>
              <div className="eyebrow">SENSOR TELEMETRY</div>
              <h2>Evidence Signals</h2>
            </div>
            <span className="panel-live">● CASE DATA</span>
          </div>
          <div className="telemetry-list">
            {incidentEvidence.map((item) => (
              <TelemetryRow
                key={item.id}
                label={item.source}
                value={item.value}
                percentage={item.reliability}
                state={item.severity.toUpperCase()}
              />
            ))}
          </div>
        </section>

        <section className="command-panel reconstruction-panel">
          <div className="command-panel-header">
            <div>
              <div className="eyebrow">TEMPORAL RECONSTRUCTION</div>
              <h2>Incident Sequence</h2>
            </div>
            <button className="panel-link" onClick={onOpenIncidents}>View investigation →</button>
          </div>
          <div className="incident-sequence">
            {sequence.map((item, index) => (
              <SequenceEvent
                key={item.id}
                time={item.time}
                title={item.title}
                source={item.source}
                severity={item.severity}
                first={index === 0}
                last={index === sequence.length - 1}
              />
            ))}
          </div>
        </section>

        <section className="command-panel conflict-panel-new">
          <div className="command-panel-header">
            <div>
              <div className="eyebrow">EVIDENCE AGREEMENT</div>
              <h2>Conflict Analysis</h2>
            </div>
            <span className="panel-live">DS K {conflictPercent.toFixed(1)}%</span>
          </div>
          {conflicts.length === 0 ? (
            <div className="empty-state">No manually recorded conflicts for this case.</div>
          ) : (
            <div className="conflict-list-new">
              {conflicts.slice(0, 3).map((conflict) => (
                <div className="conflict-row-new" key={conflict.id}>
                  <div className="conflict-row-indicator" />
                  <div>
                    <strong>{conflict.title}</strong>
                    <span>{conflict.sources}</span>
                  </div>
                  <SeverityBadge severity={conflict.severity} />
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="command-panel pipeline-panel">
          <div className="command-panel-header">
            <div>
              <div className="eyebrow">RESEARCH PIPELINE</div>
              <h2>Evidence Processing</h2>
            </div>
          </div>
          <div className="pipeline-row">
            <PipelineNode number="01" title="Collect" text={`${incidentEvidence.length} sources`} />
            <PipelineConnector />
            <PipelineNode number="02" title="Align" text="Temporal" />
            <PipelineConnector />
            <PipelineNode number="03" title="Discount" text="Reliability" />
            <PipelineConnector />
            <PipelineNode number="04" title="Fuse" text="D-S Theory" />
            <PipelineConnector />
            <PipelineNode number="05" title="Explain" text="Failure analysis" active />
          </div>
        </section>
      </div>
    </div>
  );
}

function TelemetryRow({
  label,
  value,
  percentage,
  state,
}: {
  label: string;
  value: string;
  percentage: number;
  state: string;
}) {
  return (
    <div className="telemetry-row">
      <div className="telemetry-name">
        <span>{label}</span>
        <small>{state}</small>
      </div>

      <div className="telemetry-track">
        <span
          style={{
            width: `${percentage}%`,
          }}
        />
      </div>

      <strong>{value}</strong>
    </div>
  );
}

function SequenceEvent({
  time,
  title,
  source,
  severity,
  first,
  last,
}: {
  time: string;
  title: string;
  source: string;
  severity: Severity;
  first?: boolean;
  last?: boolean;
}) {
  return (
    <div
      className={`sequence-event ${
        first ? "first" : ""
      } ${last ? "last" : ""}`}
    >
      <div className="sequence-time">
        {time}
      </div>

      <div className="sequence-marker-wrap">
        <div
          className={`sequence-marker ${severity}`}
        />
      </div>

      <div className="sequence-content">
        <strong>{title}</strong>
        <span>{source}</span>
      </div>
    </div>
  );
}

function PipelineNode({
  number,
  title,
  text,
  active,
}: {
  number: string;
  title: string;
  text: string;
  active?: boolean;
}) {
  return (
    <div
      className={`research-node ${
        active ? "active" : ""
      }`}
    >
      <div className="research-node-number">
        {number}
      </div>

      <strong>{title}</strong>

      <span>{text}</span>
    </div>
  );
}

function PipelineConnector() {
  return (
    <div className="research-connector">
      <span />
    </div>
  );
}

function MetaItem({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="meta-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

/* =========================================================
   INCIDENTS
========================================================= */

/* =========================================================
   INCIDENTS
========================================================= */

function IncidentsPage({
  selectedIncident,
  investigationTab,
  onSelectIncident,
  onBack,
  onTabChange,
  selectedEvidenceId,
  setSelectedEvidenceId,
}: {
  selectedIncident: Incident;
  investigationTab: InvestigationTab;
  onSelectIncident: (id: string) => void;
  onBack: () => void;
  onTabChange: (
    tab: InvestigationTab,
    incidentId?: string
  ) => void;
  selectedEvidenceId: string;
  setSelectedEvidenceId: (id: string) => void;
}) {
  const { evidence: evidenceData, incidents, conflicts } = useResearchCase();
  const activeCount = incidents.filter(
    (incident) =>
      incident.status === "active" ||
      incident.status === "investigating"
  ).length;

  const criticalCount = incidents.filter(
    (incident) => incident.severity === "critical"
  ).length;

  const averageConfidence =
    incidents.length > 0
      ? incidents.reduce(
          (sum, incident) => sum + incident.confidence,
          0
        ) / incidents.length
      : 0;

  const selectedEvidence = evidenceData.filter((item) =>
    selectedIncident.evidenceIds.includes(item.id)
  );

  return (
    <div className="page">
      {/* ---------------------------------------------------
          PAGE HEADER
      --------------------------------------------------- */}

      <PageHeader
        eyebrow="INCIDENT COMMAND"
        title="Incident Investigation"
        description="Monitor active industrial incidents, inspect supporting evidence, and reconstruct failure progression."
      />

      {/* ---------------------------------------------------
          INCIDENT METRICS
      --------------------------------------------------- */}

      <div className="stats-grid">
        <StatCard
          label="Tracked Incidents"
          value={String(incidents.length).padStart(2, "0")}
          meta="Current investigation set"
        />

        <StatCard
          label="Active Cases"
          value={String(activeCount).padStart(2, "0")}
          meta="Requires investigation"
        />

        <StatCard
          label="Critical Cases"
          value={String(criticalCount).padStart(2, "0")}
          meta="Immediate attention"
        />

        <StatCard
          label="Mean Confidence"
          value={`${averageConfidence.toFixed(1)}%`}
          meta="Across tracked incidents"
        />
      </div>

      {/* ---------------------------------------------------
          INCIDENT COMMAND BOARD
      --------------------------------------------------- */}

      <div className="incident-layout">
        {/* -------------------------------------------------
            CASE RAIL
        ------------------------------------------------- */}

        <section className="panel incident-list">
          <div className="panel-heading">
            <div>
              <div className="eyebrow">
                CASE CONTROL
              </div>

              <h2>Tracked incidents</h2>

              <p className="muted">
                Select an incident to inspect its evidence chain.
              </p>
            </div>

            <span className="panel-count">
              {incidents.length}
            </span>
          </div>

          <div className="incident-list-stack">
            {incidents.map((incident) => {
              const isSelected =
                selectedIncident.id === incident.id;

              const incidentEvidence =
                evidenceData.filter((item) =>
                  incident.evidenceIds.includes(item.id)
                );

              return (
                <button
                  key={incident.id}
                  className={`incident-list-item ${
                    isSelected ? "selected" : ""
                  }`}
                  onClick={() =>
                    onSelectIncident(incident.id)
                  }
                >
                  <div className="incident-list-main">
                    <div className="incident-list-top">
                      <span className="incident-list-id">
                        {incident.id}
                      </span>

                      <SeverityBadge
                        severity={incident.severity}
                      />
                    </div>

                    <strong className="incident-list-title">
                      {incident.title}
                    </strong>

                    <span className="incident-list-equipment">
                      {incident.equipment}
                    </span>

                    <div className="incident-list-meta">
                      <span>
                        {incident.started}
                      </span>

                      <span>
                        {incidentEvidence.length} evidence
                      </span>
                    </div>
                  </div>

                  <div className="incident-list-confidence">
                    <strong>
                      {incident.confidence}%
                    </strong>

                    <span>
                      confidence
                    </span>

                    <div className="small-progress">
                      <span
                        style={{
                          width: `${incident.confidence}%`,
                        }}
                      />
                    </div>
                  </div>

                  <span className="row-arrow">
                    →
                  </span>
                </button>
              );
            })}
          </div>
        </section>

        {/* -------------------------------------------------
            SELECTED INCIDENT
        ------------------------------------------------- */}

        <section className="panel incident-command-detail">
          <div className="investigation-header">
            <div className="investigation-heading">
              <div className="eyebrow">
                SELECTED CASE · {selectedIncident.id}
              </div>

              <h1>
                {selectedIncident.equipment}
                {" · "}
                {selectedIncident.title}
              </h1>

              <p className="muted">
                {selectedIncident.summary}
              </p>
            </div>

            <div className="investigation-status">
              <StatusBadge
                status={selectedIncident.status}
              />

              <SeverityBadge
                severity={selectedIncident.severity}
              />
            </div>
          </div>

          {/* -----------------------------------------------
              INCIDENT SNAPSHOT
          ----------------------------------------------- */}

          <div className="incident-snapshot">
            <div className="snapshot-item">
              <span>STARTED</span>
              <strong>
                {selectedIncident.started}
              </strong>
            </div>

            <div className="snapshot-item">
              <span>LOCATION</span>
              <strong>
                {selectedIncident.location}
              </strong>
            </div>

            <div className="snapshot-item">
              <span>EVIDENCE</span>
              <strong>
                {selectedEvidence.length}
              </strong>
            </div>

            <div className="snapshot-item">
              <span>CONFIDENCE</span>
              <strong>
                {selectedIncident.confidence}%
              </strong>
            </div>

            <div className="snapshot-item">
              <span>ROOT CAUSE</span>
              <strong>
                {selectedIncident.rootCause}
              </strong>
            </div>
          </div>

          {/* -----------------------------------------------
              INVESTIGATION TABS
          ----------------------------------------------- */}

          <div className="investigation-tabs">
            {(
              [
                "overview",
                "evidence",
                "timeline",
                "conflicts",
                "analysis",
              ] as InvestigationTab[]
            ).map((tab) => (
              <button
                key={tab}
                className={
                  investigationTab === tab
                    ? "active"
                    : ""
                }
                onClick={() =>
                  onTabChange(
                    tab,
                    selectedIncident.id
                  )
                }
              >
                {tab === "overview"
                  ? "Overview"
                  : tab === "evidence"
                  ? "Evidence"
                  : tab === "timeline"
                  ? "Timeline"
                  : tab === "conflicts"
                  ? "Conflicts"
                  : "Failure Analysis"}
              </button>
            ))}
          </div>

          {/* -----------------------------------------------
              INVESTIGATION CONTENT
          ----------------------------------------------- */}

          <div className="investigation-content">
            {investigationTab === "overview" && (
              <IncidentOverview
                incident={selectedIncident}
              />
            )}

            {investigationTab === "evidence" && (
              <IncidentEvidence
                incident={selectedIncident}
                selectedEvidenceId={
                  selectedEvidenceId
                }
                setSelectedEvidenceId={
                  setSelectedEvidenceId
                }
              />
            )}

            {investigationTab === "timeline" && (
              <IncidentTimeline
                incident={selectedIncident}
              />
            )}

            {investigationTab === "conflicts" && (
              <IncidentConflicts
                incident={selectedIncident}
              />
            )}

            {investigationTab === "analysis" && (
              <IncidentAnalysis
                incident={selectedIncident}
              />
            )}
          </div>

          {/* -----------------------------------------------
              FOOTER ACTION
          ----------------------------------------------- */}

          <div className="investigation-footer">
            <button
              className="secondary-button"
              onClick={onBack}
            >
              ← Incident Command
            </button>

            <div className="investigation-footer-status">
              <span className="online-dot" />

              <span>
                Evidence chain synchronized
              </span>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

/* =========================================================
   INCIDENT OVERVIEW
========================================================= */

function IncidentOverview({
  incident,
}: {
  incident: Incident;
}) {
  const { evidence: evidenceData, incidents, conflicts } = useResearchCase();
  return (
    <div className="overview-grid">
      <div className="panel">
        <div className="eyebrow">
          INCIDENT SUMMARY
        </div>

        <h2>{incident.rootCause}</h2>

        <p>{incident.summary}</p>
        <div className="incident-summary-grid">
          <div>
            <span>Status</span>
            <strong>{incident.status}</strong>
          </div>

          <div>
            <span>Started</span>
            <strong>{incident.started}</strong>
          </div>

          <div>
            <span>Location</span>
            <strong>{incident.location}</strong>
          </div>

          <div>
            <span>Evidence</span>
            <strong>
              {incident.evidenceIds.length}
            </strong>
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="eyebrow">
          EVIDENCE COVERAGE
        </div>

        {evidenceData
          .filter((item) =>
            incident.evidenceIds.includes(
              item.id
            )
          )
          .map((item) => (
            <MetricBar
              key={item.id}
              label={item.id}
              value={item.reliability}
            />
          ))}
      </div>
    </div>
  );
}

/* =========================================================
   INCIDENT EVIDENCE
========================================================= */

function IncidentEvidence({
  incident,
  selectedEvidenceId,
  setSelectedEvidenceId,
}: {
  incident: Incident;
  selectedEvidenceId: string;
  setSelectedEvidenceId: (
    id: string
  ) => void;
}) {
  const { evidence: evidenceData, incidents, conflicts } = useResearchCase();
  const items = evidenceData.filter((item) =>
    incident.evidenceIds.includes(item.id)
  );

  const selected =
    items.find(
      (item) => item.id === selectedEvidenceId
    ) ?? items[0];

  return (
    <div className="evidence-workspace">
      <div className="panel evidence-list-panel">
        <div className="eyebrow">
          {items.length} ITEMS
        </div>

        {items.map((item) => (
          <button
            key={item.id}
            className={`evidence-item ${
              selected?.id === item.id
                ? "selected"
                : ""
            }`}
            onClick={() =>
              setSelectedEvidenceId(item.id)
            }
          >
            <div className="evidence-item-top">
              <span>{item.id}</span>
              <SeverityBadge
                severity={item.severity}
              />
            </div>

            <div className="evidence-item-title">
              {item.title}
            </div>

            <div className="evidence-item-source">
              {item.source}
            </div>

            <div className="evidence-item-time">
              {item.time}
            </div>
          </button>
        ))}
      </div>

      {selected && (
        <div className="panel evidence-detail">
          <div className="eyebrow">
            SELECTED EVIDENCE
          </div>

          <div className="detail-title-row">
            <div>
              <h2>{selected.title}</h2>

              <div className="detail-id">
                {selected.id} • {selected.source}
              </div>
            </div>

            <SeverityBadge
              severity={selected.severity}
            />
          </div>

          <div className="evidence-value">
            {selected.value}
          </div>

          <p className="muted">
            {selected.description}
          </p>

          <MetricBar
            label="Source Reliability"
            value={selected.reliability}
          />

          <MetricBar
            label="Evidence Confidence"
            value={selected.confidence}
          />
        </div>
      )}
    </div>
  );
}

/* =========================================================
   INCIDENT TIMELINE
========================================================= */

function IncidentTimeline({
  incident,
}: {
  incident: Incident;
}) {
  const { evidence: evidenceData, incidents, conflicts } = useResearchCase();
  const items = evidenceData
    .filter((item) =>
      incident.evidenceIds.includes(item.id)
    )
    .sort((a, b) =>
      a.time.localeCompare(b.time)
    );

  return (
    <div className="timeline">
      {items.map((item) => (
        <div
          className="timeline-item"
          key={item.id}
        >
          <div className="timeline-time">
            {item.time}
          </div>

          <div
            className={`timeline-marker ${item.severity}`}
          />

          <div className="timeline-card">
            <div className="event-title">
              {item.title}
            </div>

            <div className="event-source">
              {item.source} • {item.id}
            </div>

            <div className="timeline-value">
              {item.value}
            </div>

            <p>{item.description}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

/* =========================================================
   INCIDENT CONFLICTS
========================================================= */

function IncidentConflicts({
  incident,
}: {
  incident: Incident;
}) {
  const { evidence: evidenceData, conflicts } = useResearchCase();
  const items = evidenceData.filter((item) => incident.evidenceIds.includes(item.id));
  const relevantConflicts = conflicts.filter((conflict) =>
    conflict.sources
      .split(/↔|->|→/)
      .map((id) => id.trim())
      .some((id) => incident.evidenceIds.includes(id))
  );
  const result = runDempsterShaferFusion(
    items,
    items.map((item) => item.id),
    Object.fromEntries(items.map((item) => [item.id, item.reliability]))
  );
  const resolved = relevantConflicts.filter((item) => item.status === "resolved").length;
  const unresolved = relevantConflicts.length - resolved;
  const primaryConflict = relevantConflicts[0];
  const sourceA = primaryConflict?.sources.split(/↔|->|→/)[0]?.trim();
  const sourceB = primaryConflict?.sources.split(/↔|->|→/)[1]?.trim();
  const evidenceA = items.find((item) => item.id === sourceA);
  const evidenceB = items.find((item) => item.id === sourceB);

  return (
    <div className="conflicts-page">
      <div className="conflict-status-strip">
        <div className="conflict-status-card">
          <span>TOTAL CONFLICTS</span>
          <strong>{String(relevantConflicts.length).padStart(2, "0")}</strong>
          <small>Detected in selected incident</small>
        </div>
        <div className="conflict-status-card resolved">
          <span>RESOLVED</span>
          <strong>{String(resolved).padStart(2, "0")}</strong>
          <small>Reliability-informed decisions</small>
        </div>
        <div className="conflict-status-card unresolved">
          <span>UNRESOLVED</span>
          <strong>{String(unresolved).padStart(2, "0")}</strong>
          <small>Requires additional evidence</small>
        </div>
        <div className="conflict-status-card critical">
          <span>DS CONFLICT K</span>
          <strong>{(result.conflict * 100).toFixed(1)}%</strong>
          <small>Calculated from incident evidence</small>
        </div>
      </div>

      <section className="conflict-overview">
        <div className="conflict-overview-main">
          <div className="eyebrow">EVIDENCE CONFLICT ANALYSIS · {incident.id}</div>
          <h2>{primaryConflict?.title ?? "No direct conflict recorded"}</h2>
          <p>
            {primaryConflict?.description ??
              "The selected incident has no manually registered source disagreement. The fusion engine continues to evaluate evidence compatibility through Dempster-Shafer conflict mass."}
          </p>

          {primaryConflict && (
            <div className="conflict-flow">
              <div className="conflict-flow-node">
                <div className="flow-node-icon">A</div>
                <strong>{evidenceA?.source ?? sourceA}</strong>
                <small>{sourceA}</small>
              </div>
              <div className={`conflict-flow-line ${primaryConflict.status === "resolved" ? "resolved" : ""}`}>
                <span>{primaryConflict.status.toUpperCase()}</span>
              </div>
              <div className="conflict-flow-node observation">
                <div className="flow-node-icon">B</div>
                <strong>{evidenceB?.source ?? sourceB}</strong>
                <small>{sourceB}</small>
              </div>
              <div className={`conflict-flow-line ${primaryConflict.status === "resolved" ? "resolved" : ""}`}>
                <span>FUSE</span>
              </div>
              <div className="conflict-flow-node decision">
                <div className="flow-node-icon">DS</div>
                <strong>Fusion decision</strong>
                <small>{primaryConflict.resolution}</small>
              </div>
            </div>
          )}
        </div>

        <div className="conflict-overview-score">
          <div
            className="conflict-score-ring"
            style={{
              background: `conic-gradient(#4bd8a5 ${Math.min(100, (1 - result.conflict) * 100)}%, rgba(255,255,255,0.055) ${Math.min(100, (1 - result.conflict) * 100)}% 100%)`,
            }}
          >
            <div>
              <strong>{((1 - result.conflict) * 100).toFixed(0)}%</strong>
              <span>AGREEMENT</span>
            </div>
          </div>
          <div className="conflict-score-label">
            <span>INCIDENT EVIDENCE</span>
            <strong>{items.length} SOURCES</strong>
          </div>
        </div>
      </section>

      <div className="conflict-section-heading">
        <div>
          <div className="eyebrow">CONFLICT REGISTER</div>
          <h2>Source disagreements</h2>
          <p>Only conflicts connected to the selected incident are shown.</p>
        </div>
        <span className="conflict-register-count">{relevantConflicts.length} RECORDS</span>
      </div>

      <div className="conflict-register">
        {relevantConflicts.length === 0 ? (
          <section className="panel empty-state">No source-level conflicts are registered for this incident.</section>
        ) : relevantConflicts.map((conflict, index) => {
          const ids = conflict.sources.split(/↔|->|→/).map((id) => id.trim());
          const left = items.find((item) => item.id === ids[0]);
          const right = items.find((item) => item.id === ids[1]);
          return (
            <article className={`conflict-analysis-card ${conflict.severity} ${conflict.status}`} key={conflict.id}>
              <div className="conflict-index">{String(index + 1).padStart(2, "0")}</div>
              <div className="conflict-analysis-main">
                <div className="conflict-analysis-header">
                  <div>
                    <div className="conflict-analysis-id">{conflict.id}</div>
                    <h3>{conflict.title}</h3>
                  </div>
                  <div className="conflict-badges">
                    <span className={`conflict-status-badge ${conflict.status}`}>{conflict.status}</span>
                    <SeverityBadge severity={conflict.severity} />
                  </div>
                </div>
                <div className="conflict-source-comparison">
                  {[left, right].map((item, sourceIndex) => (
                    <div className="conflict-evidence-source" key={sourceIndex}>
                      <div className="conflict-source-header">
                        <span className="conflict-source-id">{item?.id ?? ids[sourceIndex]}</span>
                        <span>{item?.reliability ?? "—"}% REL.</span>
                      </div>
                      <strong>{item?.title ?? "Evidence source"}</strong>
                      <div className="conflict-source-value">{item?.value ?? "Not available in selected incident"}</div>
                    </div>
                  ))}
                  <div className="conflict-source-middle">VS</div>
                </div>
                <p className="conflict-analysis-description">{conflict.description}</p>
                <div className="conflict-resolution">
                  <span>FUSION DECISION</span>
                  <strong>{conflict.resolution}</strong>
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}

function IncidentAnalysis({
  incident,
}: {
  incident: Incident;
}) {
  const { evidence: evidenceData } = useResearchCase();
  const items = evidenceData
    .filter((item) => incident.evidenceIds.includes(item.id))
    .sort((a, b) => a.time.localeCompare(b.time));
  const result = runDempsterShaferFusion(
    items,
    items.map((item) => item.id),
    Object.fromEntries(items.map((item) => [item.id, item.reliability]))
  );
  const primary = result.diagnostics[0];
  const confidence = primary?.pignistic ?? 0;

  return (
    <div className="failure-analysis-page">
      <section className="analysis-identity">
        <div>
          <div className="eyebrow">FAILURE RECONSTRUCTION · {incident.id}</div>
          <h2>{incident.equipment} · {incident.title}</h2>
          <p>{incident.summary}</p>
        </div>
        <div className="analysis-status">
          <span className="analysis-live-dot" />
          <span>{incident.status.toUpperCase()}</span>
          <i />
          <span>{items.length} EVIDENCE SOURCES</span>
        </div>
        <div className="analysis-identity-right">
          <span>FUSED CONFIDENCE</span>
          <strong>{(confidence * 100).toFixed(1)}%</strong>
        </div>
      </section>

      <section className="analysis-primary">
        <div className="analysis-primary-content">
          <div className="eyebrow">PRIMARY ROOT CAUSE</div>
          <div className="analysis-primary-layout">
            <div className="analysis-confidence">
              <div
                className="analysis-confidence-ring"
                style={{
                  background: `conic-gradient(#4bd8a5 ${confidence * 100}%, rgba(255,255,255,0.05) ${confidence * 100}% 100%)`,
                }}
              >
                <div>
                  <strong>{(confidence * 100).toFixed(1)}%</strong>
                  <span>PIGNISTIC</span>
                </div>
              </div>
              <div className="analysis-confidence-label">
                <span>FUSION STATUS</span>
                <strong>{confidence >= 0.75 ? "HIGH" : confidence >= 0.5 ? "MODERATE" : "LOW"}</strong>
              </div>
            </div>
            <div className="analysis-root-cause">
              <div className="analysis-hypothesis-label">HIGHEST RANKED HYPOTHESIS</div>
              <h2>{primary?.label ?? "No hypothesis"}</h2>
              <p>
                The Dempster-Shafer result ranks this hypothesis highest after reliability discounting and evidence combination for the selected incident.
              </p>
              <div className="analysis-root-metrics">
                <div><span>BELIEF</span><strong>{((primary?.belief ?? 0) * 100).toFixed(1)}%</strong></div>
                <div><span>PLAUSIBILITY</span><strong>{((primary?.plausibility ?? 0) * 100).toFixed(1)}%</strong></div>
                <div><span>CONFLICT K</span><strong>{(result.conflict * 100).toFixed(1)}%</strong></div>
              </div>
            </div>
          </div>
        </div>
        <div className="analysis-primary-verdict">
          <span>ANALYTICAL VERDICT</span>
          <strong>{incident.rootCause}</strong>
          <small>Evidence-fusion interpretation, not a calibrated physical probability.</small>
        </div>
      </section>

      <section className="analysis-chain-panel">
        <div className="analysis-section-header">
          <div>
            <div className="eyebrow">EXPLAINABLE EVIDENCE CHAIN</div>
            <h2>How the incident developed</h2>
            <p>Chronological evidence retained from the selected incident.</p>
          </div>
          <span className="analysis-chain-count">{items.length} EVENTS</span>
        </div>
        <div className="analysis-chain-visual">
          {items.map((item, index) => (
            <div className={`analysis-chain-event ${item.severity}`} key={item.id}>
              <div className="analysis-chain-number">{String(index + 1).padStart(2, "0")}</div>
              <div className="analysis-chain-node"><span /></div>
              <div className="analysis-chain-card">
                <div className="analysis-chain-card-top">
                  <div><span>{item.time} · {item.id}</span><strong>{item.title}</strong></div>
                  <span>{item.confidence}% confidence</span>
                </div>
                <div className="analysis-chain-source"><span>{item.source}</span><span>•</span><span>{item.value}</span></div>
                <p>{item.description}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="analysis-hypotheses">
        <div className="analysis-section-header">
          <div>
            <div className="eyebrow">FUSED FAILURE HYPOTHESES</div>
            <h2>Competing explanations</h2>
            <p>Belief, plausibility and pignistic probability after evidence fusion.</p>
          </div>
        </div>
        <div className="analysis-hypothesis-list">
          {result.diagnostics.map((diagnostic, index) => (
            <div className={`analysis-hypothesis ${index === 0 ? "primary" : ""}`} key={diagnostic.hypothesis}>
              <div className="analysis-hypothesis-rank">0{index + 1}</div>
              <div className="analysis-hypothesis-body">
                <div className="analysis-hypothesis-top">
                  <div><strong>{diagnostic.label}</strong><span>Belief {((diagnostic.belief) * 100).toFixed(1)}% · Plausibility {((diagnostic.plausibility) * 100).toFixed(1)}%</span></div>
                  <strong>{(diagnostic.pignistic * 100).toFixed(1)}%</strong>
                </div>
                <div className="analysis-probability-bar"><span style={{ width: `${diagnostic.pignistic * 100}%` }} /></div>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="analysis-evidence-panel">
        <div className="analysis-section-header">
          <div><div className="eyebrow">EVIDENCE SUPPORT</div><h2>Reliability contribution</h2><p>Source reliability used during the fusion calculation.</p></div>
        </div>
        {items.map((item) => <MetricBar key={item.id} label={`${item.id} · ${item.source}`} value={item.reliability} />)}
      </section>

      <section className="analysis-conclusion">
        <div className="analysis-section-header">
          <div><div className="eyebrow">ANALYTICAL CONCLUSION</div><h2>Research interpretation</h2></div>
        </div>
        <p>The fused evidence currently favors <strong>{primary?.label}</strong> with a pignistic probability of <strong>{(confidence * 100).toFixed(1)}%</strong>.</p>
        <p className="muted">This is an evidence-fusion decision measure rather than a directly observed physical probability. Historical calibration and additional validation are required for a research-grade predictive model.</p>
      </section>
    </div>
  );
}

function EvidenceExplorer({
  selectedEvidenceId,
  setSelectedEvidenceId,
}: {
  selectedEvidenceId: string;
  setSelectedEvidenceId: (
    id: string
  ) => void;
}) {
  const { evidence: evidenceData, incidents, conflicts } = useResearchCase();
  const sources = [
    "Temperature Sensor",
    "Pressure Sensor",
    "Vibration Sensor",
    "System Logs",
    "Maintenance Report",
    "Operator Report",
  ] as EvidenceSource[];

  const [activeSources, setActiveSources] =
    useState<EvidenceSource[]>(sources);

  const filteredEvidence =
    evidenceData.filter((item) =>
      activeSources.includes(item.source)
    );

  const selected =
    filteredEvidence.find(
      (item) => item.id === selectedEvidenceId
    ) ?? filteredEvidence[0];

  const toggleSource = (
    source: EvidenceSource
  ) => {
    setActiveSources((current) =>
      current.includes(source)
        ? current.filter(
            (item) => item !== source
          )
        : [...current, source]
    );
  };

  return (
    <div className="page">
      <PageHeader
        eyebrow="EVIDENCE"
        title="Evidence Explorer"
        description="Inspect individual evidence items and their reliability and confidence."
      />

      <div className="evidence-workspace">
        <div className="panel source-panel">
          <div className="eyebrow">SOURCES</div>

          {sources.map((source) => (
            <label
              className="source-option"
              key={source}
            >
              <input
                type="checkbox"
                checked={activeSources.includes(
                  source
                )}
                onChange={() =>
                  toggleSource(source)
                }
              />

              <span>{source}</span>
            </label>
          ))}
        </div>

        <div className="panel evidence-list-panel">
          <div className="eyebrow">
            {filteredEvidence.length} ITEMS
          </div>

          {filteredEvidence.map((item) => (
            <button
              key={item.id}
              className={`evidence-item ${
                selected?.id === item.id
                  ? "selected"
                  : ""
              }`}
              onClick={() =>
                setSelectedEvidenceId(item.id)
              }
            >
              <div className="evidence-item-top">
                <span>{item.id}</span>

                <SeverityBadge
                  severity={item.severity}
                />
              </div>

              <div className="evidence-item-title">
                {item.title}
              </div>

              <div className="evidence-item-source">
                {item.source}
              </div>

              <div className="evidence-item-time">
                {item.time}
              </div>
            </button>
          ))}
        </div>

        {selected && (
          <div className="panel evidence-detail">
            <div className="eyebrow">
              SELECTED EVIDENCE
            </div>

            <div className="detail-title-row">
              <div>
                <h2>{selected.title}</h2>

                <div className="detail-id">
                  {selected.id} •{" "}
                  {selected.source}
                </div>
              </div>

              <SeverityBadge
                severity={selected.severity}
              />
            </div>

            <div className="evidence-value">
              {selected.value}
            </div>

            <p className="muted">
              {selected.description}
            </p>

            <MetricBar
              label="Source Reliability"
              value={selected.reliability}
            />

            <MetricBar
              label="Evidence Confidence"
              value={selected.confidence}
            />
          </div>
        )}
      </div>
    </div>
  );
}

/* =========================================================
   TOP LEVEL TIMELINE
========================================================= */

function TimelinePage({
  incident,
  onOpenInvestigation,
}: {
  incident: Incident;
  onOpenInvestigation: () => void;
}) {
  const { evidence: evidenceData, incidents, conflicts } = useResearchCase();
  return (
    <div className="page">
      <PageHeader
        eyebrow="TEMPORAL ANALYSIS"
        title="Incident Timeline"
        description={`Chronological evidence reconstruction for ${incident.id}.`}
      />

      <button
        className="primary-button"
        onClick={onOpenInvestigation}
      >
        Open Full Investigation →
      </button>

      <div className="top-level-timeline panel">
        {evidenceData
          .filter((item) =>
            incident.evidenceIds.includes(
              item.id
            )
          )
          .sort((a, b) =>
            a.time.localeCompare(b.time)
          )
          .map((item) => (
            <div
              className="full-timeline-item"
              key={item.id}
            >
              <div className="timeline-time">
                {item.time}
              </div>

              <div
                className={`timeline-marker ${item.severity}`}
              />

              <div className="timeline-card">
                <div className="event-title">
                  {item.title}
                </div>

                <div className="event-source">
                  {item.source} • {item.id}
                </div>

                <div className="timeline-value">
                  {item.value}
                </div>
              </div>
            </div>
          ))}
      </div>
    </div>
  );
}

/* =========================================================
   TOP LEVEL CONFLICTS
========================================================= */

function ConflictsPage({
  incident,
  onOpenInvestigation,
}: {
  incident: Incident;
  onOpenInvestigation: () => void;
}) {
  return (
    <div className="page">
      <PageHeader
        eyebrow="CONFLICT INTELLIGENCE"
        title="Evidence Conflicts"
        description={`Conflict-aware source comparison for ${incident.id}.`}
      />
      <IncidentConflicts incident={incident} />
      <button className="primary-button" onClick={onOpenInvestigation}>
        Open {incident.id} Investigation →
      </button>
    </div>
  );
}

function FusionEnginePage({
  incident,
}: {
  incident: Incident;
}) {
  const { evidence: evidenceData } = useResearchCase();
  const items = evidenceData.filter((item) => incident.evidenceIds.includes(item.id));
  const [selectedIds, setSelectedIds] = useState<string[]>(items.map((item) => item.id));
  const [weights, setWeights] = useState<Record<string, number>>(Object.fromEntries(items.map((item) => [item.id, item.reliability])));

  const result = useMemo(() => runDempsterShaferFusion(items, selectedIds, weights), [items, selectedIds, weights]);
  const primary = result.diagnostics[0];
  const selectedItems = items.filter((item) => selectedIds.includes(item.id));
  const selectAll = () => setSelectedIds(items.map((item) => item.id));
  const clearAll = () => setSelectedIds([]);
  const resetWeights = () => setWeights(Object.fromEntries(items.map((item) => [item.id, item.reliability])));
  const confidence = primary?.pignistic ?? 0;

  return (
    <div className="fusion-engine-page">
      <PageHeader eyebrow="FORMAL ANALYSIS" title="Dempster-Shafer Fusion Engine" description={`Interactive evidence fusion for ${incident.id}. Adjust source reliability and observe the fused hypothesis.`} />

      <div className="fusion-engine-status">
        <div className="fusion-engine-status-card active"><span>ACTIVE INCIDENT</span><strong>{incident.id}</strong><small>{incident.equipment}</small></div>
        <div className="fusion-engine-status-card"><span>EVIDENCE SELECTED</span><strong>{selectedItems.length}<em>/{items.length}</em></strong><small>Sources included in fusion</small></div>
        <div className="fusion-engine-status-card"><span>PRIMARY CONFIDENCE</span><strong>{(confidence * 100).toFixed(1)}%</strong><small>{primary?.label}</small></div>
        <div className="fusion-engine-status-card"><span>BELIEF</span><strong>{((primary?.belief ?? 0) * 100).toFixed(1)}%</strong><small>Exact singleton support</small></div>
        <div className="fusion-engine-status-card"><span>CONFLICT K</span><strong className={result.conflict > 0.3 ? "danger" : result.conflict > 0.15 ? "warning" : ""}>{(result.conflict * 100).toFixed(1)}%</strong><small>Combined evidence conflict</small></div>
      </div>

      <section className="fusion-primary">
        <div className="fusion-primary-main">
          <div className="eyebrow">PRIMARY FUSED HYPOTHESIS</div>
          <div className="fusion-primary-grid">
            <div className="fusion-primary-score"><strong>{(confidence * 100).toFixed(1)}<span>%</span></strong><small>PIGNISTIC PROBABILITY</small></div>
            <div className="fusion-primary-divider" />
            <div className="fusion-primary-hypothesis"><span>HIGHEST RANKED HYPOTHESIS</span><h2>{primary?.label}</h2><p>Reliability-discounted evidence is combined sequentially using Dempster's rule of combination.</p><div className="fusion-primary-metrics"><div><span>BELIEF</span><strong>{((primary?.belief ?? 0) * 100).toFixed(1)}%</strong></div><div><span>PLAUSIBILITY</span><strong>{((primary?.plausibility ?? 0) * 100).toFixed(1)}%</strong></div><div><span>CONFLICT</span><strong>{(result.conflict * 100).toFixed(1)}%</strong></div></div></div>
          </div>
        </div>
        <div className="fusion-primary-ring" style={{background:`conic-gradient(#45d3ff ${confidence * 100}%, rgba(255,255,255,0.05) ${confidence * 100}% 100%)`}}><div><strong>{Math.round(confidence * 100)}%</strong><span>CONFIDENCE</span></div></div>
      </section>

      <section className="fusion-pipeline">
        <div className="fusion-section-header"><div><div className="eyebrow">FUSION PIPELINE</div><h2>Evidence processing</h2><p>Each source is discounted by reliability before combination.</p></div></div>
        <div className="fusion-process-track">
          {[['01','INGEST','Collect heterogeneous observations'],['02','DISCOUNT','Apply source reliability'],['03','COMBINE','Dempster rule of combination'],['04','DIAGNOSE','Belief / plausibility'],['05','DECIDE','Pignistic ranking']].map((step,index)=>(
            <Fragment key={step[0]}><div className="fusion-process-step"><span className="fusion-process-number">{step[0]}</span><div className="fusion-process-icon">{['I','D','Σ','B','✓'][index]}</div><strong>{step[1]}</strong><span>{step[2]}</span></div>{index < 4 && <div className="fusion-process-connector">→</div>}</Fragment>
          ))}
        </div>
      </section>

      <section className="fusion-config">
        <div className="fusion-section-header"><div><div className="eyebrow">SOURCE CONFIGURATION</div><h2>Evidence reliability controls</h2><p>Toggle evidence and modify its reliability weight.</p></div><div className="fusion-actions"><button className="secondary-button" onClick={selectAll}>Select all</button><button className="secondary-button" onClick={clearAll}>Clear</button><button className="secondary-button" onClick={resetWeights}>Reset reliability</button></div></div>
        <div className="fusion-evidence-grid">
          {items.map((item) => {
            const selected = selectedIds.includes(item.id);
            const weight = weights[item.id] ?? item.reliability;
            const ds = result.evidenceResults.find((entry) => entry.evidenceId === item.id);
            return <div className={`fusion-source-card ${selected ? 'selected' : 'disabled'}`} key={item.id}>
              <div className="fusion-source-top">
                <label className="fusion-source-check"><input type="checkbox" checked={selected} onChange={() => setSelectedIds((current) => current.includes(item.id) ? current.filter((id) => id !== item.id) : [...current, item.id])}/><span /></label>
                <div><span className="fusion-source-id">{item.id} · {item.source}</span><strong>{item.title}</strong><small>{item.time} · {item.confidence}% confidence</small></div>
                <SeverityBadge severity={item.severity}/>
              </div>
              <div className="fusion-source-value">{item.value}</div>
              <div className="fusion-source-slider"><div className="fusion-slider-header"><span>RELIABILITY WEIGHT</span><strong>{weight}%</strong></div><input type="range" min="0" max="100" value={weight} disabled={!selected} onChange={(event) => setWeights((current) => ({...current, [item.id]: Number(event.target.value)}))}/></div>
              <div className="fusion-bba-row"><span>DISCOUNTED SUPPORT</span><strong>{ds ? Object.entries(ds.discountedMass).filter(([,v]) => v > 0.01).slice(0,3).map(([k,v]) => `${k}: ${(v*100).toFixed(1)}%`).join(' · ') : 'Excluded'}</strong></div>
            </div>;
          })}
        </div>
      </section>

      <section className="fusion-hypotheses">
        <div className="fusion-section-header"><div><div className="eyebrow">HYPOTHESIS RANKING</div><h2>Competing explanations</h2><p>Current D-S diagnostic ranking for the selected evidence set.</p></div></div>
        <div className="fusion-hypothesis-list">
          {result.diagnostics.map((diagnostic,index)=><div className={`fusion-hypothesis-row ${index===0?'primary':''}`} key={diagnostic.hypothesis}><div className="fusion-hypothesis-rank">0{index+1}</div><div className="fusion-hypothesis-main"><div><strong>{diagnostic.label}</strong><span>Belief {((diagnostic.belief)*100).toFixed(1)}% · Plausibility {((diagnostic.plausibility)*100).toFixed(1)}%</span></div><strong>{(diagnostic.pignistic*100).toFixed(1)}%</strong><div className="fusion-hypothesis-bar"><span style={{width:`${diagnostic.pignistic*100}%`}}/></div></div></div>)}
        </div>
      </section>
    </div>
  );
}

function FailureAnalysisPage({
  incident,
}: {
  incident: Incident;
}) {
  return (
    <div className="page">
      <PageHeader eyebrow="FAILURE ANALYSIS" title="Explainable Failure Reconstruction" description={`Evidence-backed root-cause analysis for ${incident.id}.`} />
      <IncidentAnalysis incident={incident} />
    </div>
  );
}

/* =========================================================
   REUSABLE COMPONENTS
========================================================= */

function PageHeader({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div className="page-header">
      <div className="eyebrow">{eyebrow}</div>

      <h1>{title}</h1>

      <p>{description}</p>
    </div>
  );
}

function StatCard({
  label,
  value,
  meta,
}: {
  label: string;
  value: string;
  meta: string;
}) {
  return (
    <div className="stat-card">
      <span>{label}</span>

      <strong>{value}</strong>

      <small>{meta}</small>
    </div>
  );
}

function SeverityBadge({
  severity,
}: {
  severity: Severity;
}) {
  return (
    <span className={`severity ${severity}`}>
      {severity}
    </span>
  );
}

function MetricBar({
  label,
  value,
}: {
  label: string;
  value: number;
}) {
  return (
    <div className="metric">
      <div className="metric-top">
        <span>{label}</span>

        <strong>{value}%</strong>
      </div>

      <div className="metric-track">
        <span
          style={{
            width: `${value}%`,
          }}
        />
      </div>
    </div>
  );
}

function PipelineStep({
  number,
  label,
  status,
}: {
  number: string;
  label: string;
  status: string;
}) {
  return (
    <div className="pipeline-step">
      <span className="pipeline-number">
        {number}
      </span>

      <div>
        <strong>{label}</strong>

        <small>{status}</small>
      </div>
    </div>
  );
}

/* =========================================================
   FORMATTING HELPERS
========================================================= */

function formatMass(mass: MassMap): string {
  const entries = Object.entries(mass)
    .filter(([, value]) => value > 0.001)
    .sort((a, b) => b[1] - a[1]);

  return entries
    .map(([key, value]) => {
      const label =
        key === THETA
          ? "Θ"
          : splitSet(key)
              .map(                (member) =>
                  DS_HYPOTHESES.find(
                    (hypothesis) =>
                      hypothesis.key ===
                      member
                  )?.label ?? member
              )
              .join(" + ");

      return `${label}: ${(value * 100).toFixed(
        1
      )}%`;
    })
    .join(" · ");
}

export default App;