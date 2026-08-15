from ground_truth_client.client import GroundTruthClient


client = GroundTruthClient(
    base_url="http://127.0.0.1:8300"
)


# 1. 서버 상태 확인
print("Health:")
print(client.health())


# 2. 논문 1개 테스트
result = client.label(
    profile="""
    로봇 매니퓰레이션 중에서도 grasping(파지)에 관심이 있다. 
    특히 tactile sensor와 force control을 이용해 로봇이 물체를 안정적으로 잡는 방법과, 
    사람 손의 grasping 메커니즘을 로봇에 구현하는 방식에 흥미가 있다. Learning-based 접근보다는 
    optimization 기반 방법론(contact mechanics, force closure, grasp planning의 최적화적 접근)을 선호한다.
    """,
    title="MIDAS Hand: Modular low-Impedance Direct-drive Anthropomorphic Sensing Hand",
    abstract="""
    Dexterous manipulation is limited not only by algorithms but by a shortage of accessible 
    hand hardware that combines human-scale morphology, ease of manufacturing or maintenance, 
    tactile sensing, and practical cost. Existing dexterous hands tend to optimize some of these properties 
    at the expense of others. We present MIDAS Hand, a low-cost, open-source, human-scale dexterous hand 
    with integrated tactile sensing for manipulation research. MIDAS Hand provides 16 total degrees of freedom 
    (DoF) with 13 active DoF, directly driven actuation with measurably low backdrive torque, 
    and 283 three-axis tactile taxels in a compact 700 g package with a bill of materials under 3,000 USD. 
    Built from 3D-printed components, it assembles in under three hours while providing the strength, 
    repeatability, and maintainability needed for repeated real-world experiments. Alongside the hardware, 
    we release a full stack: design files, build documentation, control and tactile Python APIs, 
    simulation models, and retargeting and teleoperation pipelines. We characterize MIDAS Hand through workspace 
    and grasp-taxonomy analysis, payload and reliability tests, backdrivability measurements, and teleoperation 
    demonstrations with tactile sensing, showing that it offers a balanced, reproducible platform 
    for tactile dexterous manipulation and human-to-robot data collection. 
    Project page: https://midas-hand.com
    """
)

print("\nLabel Result:")
print(result)