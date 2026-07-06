import folium
from folium.plugins import Fullscreen, MeasureControl, MiniMap

from LanePlanner import (
    downsample_evenly,
    latlon,
    latlon_of,
    polyline_length_m
)


# =============================================================================
# 地图渲染
# =============================================================================
def build_route_map(sn, task_id, bundle, planner, settings, show_lanes):
    task = bundle["task"]
    plan = bundle["plan"]
    history = list(task["history_points"])
    history_for_map = downsample_evenly(history, settings["max_history_points"])
    planned_points = planner.visible_points(plan)

    history_distance = polyline_length_m(history)
    planned_distance = plan["metrics"]["total_distance_m"]

    lats = [point["lat"] for point in history_for_map] + [point["lat"] for point in planned_points]
    lons = [point["lon"] for point in history_for_map] + [point["lon"] for point in planned_points]
    if not lats or not lons:
        raise RuntimeError("没有可绘制的坐标。")

    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    map_object = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True,
    )

    if show_lanes:
        add_lane_background(map_object, planner.lanes.values())

    add_history_layer(map_object, history_for_map, len(history))
    add_planned_layer(map_object, planner, plan)
    add_marker_layer(map_object, bundle)

    MiniMap(toggle_display=True).add_to(map_object)
    Fullscreen().add_to(map_object)
    MeasureControl(primary_length_unit="meters").add_to(map_object)
    folium.LayerControl(collapsed=False).add_to(map_object)
    fit_bounds(map_object, lats, lons)
    add_legend(map_object, sn, task_id, history_distance, planned_distance, len(plan["path"]))

    summary = {
        "sn": sn,
        "task_id": task_id,
        "source_file": task["source_file"].name,
        "od_source": bundle["od_source"],
        "history_point_count": len(history),
        "history_distance_m": history_distance,
        "planned_distance_m": planned_distance,
        "distance_difference_m": history_distance - planned_distance,
        "start_lane_uid": plan["start_uid"],
        "end_lane_uid": plan["end_uid"],
        "planned_lane_count": len(plan["path"]),
        "planned_lane_uids": list(plan["path"]),
        "tried_count": plan["tried_count"],
        "objective_m": plan["metrics"]["objective_m"],
        "snap_distance_m": plan["metrics"]["snap_distance_m"],
        "jump_distance_m": plan["metrics"]["jump_distance_m"],
    }

    return map_object.get_root().render(), summary


def add_lane_background(map_object, lanes):
    group = folium.FeatureGroup(name="全部车道（背景）", show=True)
    for lane in lanes:
        points = downsample_evenly(lane["points"], 100)
        folium.PolyLine(
            locations=latlon(points),
            color="#98A2B3",
            weight=1,
            opacity=0.30,
            tooltip=f"{lane.get('name')} | UID={lane.get('uid')}",
        ).add_to(group)
    group.add_to(map_object)


def add_history_layer(map_object, history, original_count):
    group = folium.FeatureGroup(name="历史路线", show=True)
    folium.PolyLine(
        locations=[latlon_of(point) for point in history],
        color="#1677ff",
        weight=5,
        opacity=0.88,
        tooltip="历史路线：%d 个原始点" % original_count,
    ).add_to(group)
    group.add_to(map_object)


def add_planned_layer(map_object, planner, plan):
    group = folium.FeatureGroup(name="规划路线", show=True)
    path = plan["path"]

    for index, uid in enumerate(path, start=1):
        lane = planner.lanes[uid]
        visible = planner.visible_lane_points(
            path,
            uid,
            index - 1,
            plan["origin_projection"],
            plan["destination_projection"],
        )
        color = "#f5222d"
        if lane["type_id"] != 101:
            color = "#fa8c16"

        popup = (
            "<b>规划路线第 %d 段</b><br>"
            "UID：%s<br>"
            "名称：%s<br>"
            "长度：%.1f m"
        ) % (index, uid, lane.get("name"), lane["length_m"])

        folium.PolyLine(
            locations=latlon(visible),
            color=color,
            weight=7,
            opacity=0.93,
            popup=folium.Popup(popup, max_width=360),
            tooltip="规划 %d: %s" % (index, lane.get("name")),
        ).add_to(group)

    group.add_to(map_object)


def add_marker_layer(map_object, bundle):
    group = folium.FeatureGroup(name="起终点与吸附点", show=True)

    origin = bundle["origin"]
    destination = bundle["destination"]
    plan = bundle["plan"]

    folium.Marker(
        latlon_of(origin),
        tooltip="规划起点",
        popup="规划起点（%s）" % bundle["od_source"],
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(group)

    folium.Marker(
        latlon_of(destination),
        tooltip="规划终点",
        popup="规划终点（%s）" % bundle["od_source"],
        icon=folium.Icon(color="blue", icon="flag"),
    ).add_to(group)

    marker_items = [
        ("起点吸附", origin, plan["origin_projection"], "#13c2c2"),
        ("终点吸附", destination, plan["destination_projection"], "#722ed1"),
    ]

    for label, point, projection, color in marker_items:
        projected = (projection["projected_lat"], projection["projected_lon"])
        folium.CircleMarker(
            projected,
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=1,
            tooltip=label,
        ).add_to(group)
        folium.PolyLine(
            [latlon_of(point), projected],
            color=color,
            weight=3,
            dash_array="6,6",
            opacity=0.9,
        ).add_to(group)

    group.add_to(map_object)


def fit_bounds(map_object, lats, lons):
    padding = 0.001
    map_object.fit_bounds([
        [min(lats) - padding, min(lons) - padding],
        [max(lats) + padding, max(lons) + padding],
    ])


def add_legend(map_object, sn, task_id, history_distance, planned_distance, lane_count):
    legend = """
    <div style="position:fixed;left:25px;bottom:25px;z-index:9999;
                background:white;border:1px solid #999;border-radius:8px;
                padding:11px 14px;font-size:13px;line-height:1.55;
                box-shadow:2px 2px 8px rgba(0,0,0,.22);max-width:360px;">
      <b>历史路线与规划路线</b><br>
      SN：%s<br>
      task_id：%s<br>
      <span style="color:#1677ff;">━━</span> 历史路线<br>
      <span style="color:#f5222d;">━━</span> 规划路线<br>
      历史路线：%.1f m<br>
      规划路线：%.1f m<br>
      规划车道数：%d
    </div>
    """ % (
        sn,
        task_id,
        history_distance,
        planned_distance,
        lane_count,
    )
    map_object.get_root().html.add_child(folium.Element(legend))
