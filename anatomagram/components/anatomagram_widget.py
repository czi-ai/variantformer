"""AnyWidget implementation for anatomogram visualization - Multi-View Widget Only."""

import anywidget
import traitlets
from pathlib import Path
import json



# AnatomagramWidget class removed in Step 5 - use AnatomagramMultiViewWidget instead
# The multi-view widget supports both multi-view (tabs) and single-view (sex parameter) modes

class AnatomagramMultiViewWidget(anywidget.AnyWidget):
    """Multi-view anatomagram widget with built-in view switching and shared caching.

    This widget can display multiple anatomogram views (male, female, brain) with
    a single widget instance that shares cached SVG content across all views.
    Designed for optimal performance in tabbed interfaces.
    """

    _version = "0.5.0"  # Step 6: Fully consolidated widget (merged BaseAnatomagramWidget)

    # === Shared Synchronized Properties (from BaseAnatomagramWidget) ===

    # Data properties (generalized format)
    visualization_data = traitlets.Dict({}).tag(sync=True)
    selected_item = traitlets.Unicode("").tag(sync=True)
    data_type = traitlets.Unicode("expression").tag(sync=True)

    # Visualization settings
    sex = traitlets.Unicode("male").tag(sync=True)
    color_palette = traitlets.Unicode("viridis").tag(sync=True)
    scale_type = traitlets.Unicode("linear").tag(sync=True)
    threshold = traitlets.Float(0.0).tag(sync=True)
    size_scale = traitlets.Float(1.0).tag(sync=True)
    no_data_opacity = traitlets.Float(0.5).tag(sync=True)

    # Mapping data
    uberon_map = traitlets.Dict({}).tag(sync=True)
    uberon_names = traitlets.Dict({}).tag(sync=True)  # Legacy name kept for compatibility
    uberon_descriptions = traitlets.Dict({}).tag(sync=True)  # PRIMARY: descriptions from uberon_descriptions.json
    enhanced_tooltips = traitlets.Dict({}).tag(sync=True)

    # SVG content (loaded from assets)
    male_svg = traitlets.Unicode("").tag(sync=True)
    female_svg = traitlets.Unicode("").tag(sync=True)
    brain_svg = traitlets.Unicode("").tag(sync=True)

    # Debug mode
    debug = traitlets.Bool(False).tag(sync=True)

    # === Legacy Properties for Backward Compatibility ===
    expression_data = traitlets.Dict({}).tag(sync=True)
    selected_gene = traitlets.Unicode("").tag(sync=True)

    # === Multi-view specific traitlets ===
    available_views = traitlets.List(
        default_value=["male", "female", "brain"],
        help="List of anatomogram views to load and cache"
    ).tag(sync=True)
    
    current_view = traitlets.Unicode(
        default_value="male",
        help="Currently displayed anatomogram view"
    ).tag(sync=True)

    # CSS styling for the widget (copied from AnatomagramWidget for independence)
    _css = """
    .anatomogram-widget-container {
        width: 100%;
        height: 60vh;      /* 60vh allows room for text above/below in web page embeds */
        position: relative;
        /* Background and text color will be set via JavaScript theme detection */
        border-radius: 8px;
        overflow: hidden;  /* prevent scrolling, JS will size to fit */
        padding: 20px;     /* add padding to prevent edge clipping */
        box-sizing: border-box;
        container-type: inline-size;  /* Enable container queries for responsive behavior */
    }

    /* Marimo-specific responsive styling */
    @container (max-width: 800px) {
        .anatomogram-widget-container {
            padding: 15px;
        }
    }

    .anatomogram-container {
        width: 100%;
        height: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        overflow: hidden; /* prevent scrolling */
    }

    .anatomogram-container svg {
        width: auto;       /* let SVG determine its own width */
        height: auto;      /* maintain aspect ratio */
        max-width: 100%;   /* fit within container */
        max-height: 100%;  /* fit within container height */
        opacity: 0;        /* default hidden, will fade in */
        transition: opacity 280ms ease-in-out;
        will-change: opacity;
    }

    .anatomogram-container svg path,
    .anatomogram-container svg circle,
    .anatomogram-container svg rect,
    .anatomogram-container svg polygon,
    .anatomogram-container svg ellipse {
        fill: #E0E0E0;
        stroke: white; /* Default light mode stroke */
        stroke-width: 0.5;
        transition: opacity 0.2s;
        cursor: pointer;
    }

    /* Dark mode fallback for SVG strokes */
    @media (prefers-color-scheme: dark) {
        .anatomogram-container svg path,
        .anatomogram-container svg circle,
        .anatomogram-container svg rect,
        .anatomogram-container svg polygon,
        .anatomogram-container svg ellipse {
            stroke: #666666;
        }
    }

    .anatomogram-container svg path:hover,
    .anatomogram-container svg circle:hover,
    .anatomogram-container svg rect:hover,
    .anatomogram-container svg polygon:hover,
    .anatomogram-container svg ellipse:hover {
        opacity: 0.8;
        stroke: #333333; /* Default hover stroke */
        stroke-width: 1;
    }

    .anatomogram-tooltip {
        position: absolute;
        display: none;
        background-color: rgba(0, 0, 0, 0.95); /* Theme applied via JS */
        color: white; /* Theme applied via JS */
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 13px;
        pointer-events: none;
        z-index: 1000;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        font-family: var(--marimo-text-font, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif);
        max-width: 320px;
        line-height: 1.4;
        border: 1px solid transparent; /* Theme applied via JS */
    }

    /* Dark mode fallback for tooltips */
    @media (prefers-color-scheme: dark) {
        .anatomogram-tooltip {
            background-color: rgba(0, 0, 0, 0.95);
            color: #ffffff;
        }
    }

    .anatomogram-tooltip::after {
        content: '';
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: rgba(0, 0, 0, 0.95) transparent transparent transparent;
    }

    .tooltip-main-tissue {
        font-weight: 600;
        margin-bottom: 6px;
        font-size: 14px;
        color: inherit;
        border-bottom: 1px solid;
        border-bottom-color: currentColor;
        opacity: 1;
        padding-bottom: 4px;
    }

    .tooltip-main-value {
        font-size: 13px;
        color: var(--tooltip-success-color, inherit);
        font-weight: 500;
        margin-bottom: 8px;
    }

    .tooltip-aggregation-header {
        font-size: 11px;
        color: var(--tooltip-info-color, inherit);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
        font-weight: 500;
    }

    .tooltip-contributing-tissue {
        font-size: 11px;
        color: inherit;
        opacity: 0.85;
        margin-left: 8px;
        margin-bottom: 2px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .tooltip-contributing-tissue .tissue-indicator {
        margin-right: 6px;
        font-weight: bold;
    }

    .tooltip-contributing-tissue .tissue-indicator.direct {
        color: var(--tooltip-success-color, inherit);
    }

    .tooltip-contributing-tissue .tissue-indicator.hierarchy {
        color: var(--tooltip-warning-color, inherit);
    }

    .tooltip-contributing-tissue .tissue-value {
        color: inherit;
        font-weight: 500;
    }

    .tooltip-no-data {
        font-size: 12px;
        color: var(--tooltip-muted-color, inherit);
        opacity: 0.7;
        font-style: italic;
    }

    .tooltip-hierarchy-mapping {
        font-size: 10px;
        color: var(--tooltip-warning-color, inherit);
        margin-top: 4px;
    }

    .tooltip-description {
        font-size: 11px;
        color: inherit;
        opacity: 0.75;
        margin-top: 6px;
        margin-bottom: 4px;
        line-height: 1.4;
        font-style: normal;
    }

    .tooltip-item-label {
        font-size: 11px;
        color: inherit;
        opacity: 0.7;
    }

    .loading-message {
        text-align: center;
        padding: 40px;
        font-size: 18px;
        color: #666; /* Theme applied via JS */
        font-family: var(--marimo-text-font, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif);
    }

    .error-message {
        text-align: center;
        padding: 40px;
        font-size: 16px;
        color: #d32f2f;
        font-family: var(--marimo-text-font, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif);
    }
    """

    def __init__(self, **kwargs):
        """Initialize multi-view widget with enhanced caching and single-view support.

        Supports both multi-view (with tabs) and single-view (no tabs) modes.
        Single-view mode is activated when 'sex' parameter is provided without 'available_views'.
        """

        # === Step A: Handle single-view mode via 'sex' parameter ===
        if 'sex' in kwargs and 'available_views' not in kwargs:
            # Single-view mode: use 'sex' to determine the single view
            sex_value = kwargs['sex']
            kwargs['available_views'] = [sex_value]
            kwargs['current_view'] = sex_value
            # Keep 'sex' in kwargs for parent class

        # === Step B: Handle multi-view parameters ===
        # Handle available_views parameter
        if 'available_views' in kwargs:
            self.available_views = kwargs['available_views']

        # Handle current_view parameter (also accept 'sex' for backward compatibility)
        if 'current_view' in kwargs:
            self.current_view = kwargs['current_view']
        elif 'sex' in kwargs:
            self.current_view = kwargs['sex']
            # Remove sex from kwargs to avoid conflict with parent class
            del kwargs['sex']

        # Set sex parameter for consistency
        kwargs['sex'] = self.current_view

        # === Step C: Legacy keyword normalization (from BaseAnatomagramWidget) ===
        if 'expression_data' in kwargs and 'visualization_data' not in kwargs:
            kwargs['visualization_data'] = kwargs['expression_data']
        if 'selected_gene' in kwargs and 'selected_item' not in kwargs:
            kwargs['selected_item'] = kwargs['selected_gene']

        # === Step D: Call parent init ===
        super().__init__(**kwargs)

        # === Step E: Load assets (from BaseAnatomagramWidget) ===
        self._load_svg_content()
        self._load_uberon_descriptions()

        # === Step F: Sync legacy properties (from BaseAnatomagramWidget) ===
        self._sync_legacy_properties()

        # === Step G: Ensure current_view is in available_views ===
        if self.current_view not in self.available_views:
            self.available_views = list(self.available_views) + [self.current_view]

    # === Helper methods (from BaseAnatomagramWidget) ===

    def _load_svg_content(self):
        """Load SVG content from local files into traitlets."""
        svg_dir = Path(__file__).parent.parent / "assets" / "svg"

        try:
            male_svg_path = svg_dir / "homo_sapiens.male.svg"
            female_svg_path = svg_dir / "homo_sapiens.female.svg"
            brain_svg_path = svg_dir / "homo_sapiens.brain.svg"

            if male_svg_path.exists():
                self.male_svg = male_svg_path.read_text(encoding='utf-8')
            else:
                print(f"Warning: Male SVG not found at {male_svg_path}")

            if female_svg_path.exists():
                self.female_svg = female_svg_path.read_text(encoding='utf-8')
            else:
                print(f"Warning: Female SVG not found at {female_svg_path}")

            if brain_svg_path.exists():
                self.brain_svg = brain_svg_path.read_text(encoding='utf-8')
            else:
                print(f"Warning: Brain SVG not found at {brain_svg_path}")

        except Exception as e:
            print(f"Error loading SVG content: {e}")
            self.male_svg = ""
            self.female_svg = ""
            self.brain_svg = ""

    def _load_uberon_descriptions(self):
        """Load UBERON labels and descriptions from JSON file into traitlets.

        CRITICAL: uberon_names should contain SHORT LABELS (e.g., "Prefrontal Cortex")
                  uberon_descriptions should contain LONG DESCRIPTIONS (detailed text)

        NOTE: Uses uberon_descriptions.json (NOT uberon_id_map.json).
        """
        data_dir = Path(__file__).parent.parent / "data"

        try:
            uberon_desc_path = data_dir / "uberon_descriptions.json"

            if uberon_desc_path.exists():
                with open(uberon_desc_path, 'r', encoding='utf-8') as f:
                    descriptions_data = json.load(f)

                # Extract LABELS (short names) for uberon_names - used in tooltips
                labels_dict = {}
                # Extract DESCRIPTIONS (long text) for uberon_descriptions
                descriptions_dict = {}

                for uberon_id, data in descriptions_data.items():
                    # Short label for tooltips and display (e.g., "Prefrontal Cortex")
                    if 'label' in data and data['label']:
                        labels_dict[uberon_id] = data['label']

                    # Long description for detailed info
                    if 'description' in data and data['description']:
                        descriptions_dict[uberon_id] = data['description']

                # uberon_names contains SHORT LABELS (not descriptions!)
                self.uberon_names = labels_dict
                # uberon_descriptions contains LONG DESCRIPTIONS
                self.uberon_descriptions = descriptions_dict
            else:
                print(f"Warning: UBERON descriptions not found at {uberon_desc_path}")
                self.uberon_descriptions = {}
                self.uberon_names = {}

        except Exception as e:
            print(f"Error loading UBERON descriptions: {e}")
            self.uberon_descriptions = {}
            self.uberon_names = {}

    def _sync_legacy_properties(self):
        """Sync legacy properties with new generalized properties."""
        # Lightweight sync: only if one is set and the other isn't
        if self.visualization_data and not self.expression_data:
            self.expression_data = self.visualization_data
        elif self.expression_data and not self.visualization_data:
            self.visualization_data = self.expression_data

        # Sync selected item/gene
        if self.selected_item and not self.selected_gene:
            self.selected_gene = self.selected_item
        elif self.selected_gene and not self.selected_item:
            self.selected_item = self.selected_gene

    def update_item(self, item: str):
        """Update the selected item (gene or risk category) programmatically."""
        data = self.visualization_data or self.expression_data
        if data and 'genes' in data:
            if item in data['genes']:
                self.selected_item = item
                self.selected_gene = item  # Keep legacy in sync
            else:
                available = list(data['genes'].keys())
                print(f"Warning: '{item}' not in available items: {available[:5]}")
        else:
            print("Warning: No data available to select from")

    # === Multi-view specific methods ===

    def switch_view(self, view: str):
        """Switch to a different anatomogram view."""
        if view not in self.available_views:
            raise ValueError(f"View '{view}' not in available views: {self.available_views}")
        
        self.current_view = view
        # Also update the sex parameter for backward compatibility
        self.sex = view
    
    def get_available_views(self):
        """Get list of available anatomogram views."""
        return list(self.available_views)
    
    def create_view_tabs(self, mo):
        """Create Marimo tabs for reactive view switching.
        
        Returns mo.ui.tabs widget that can be used in reactive cells to control
        this widget's current_view. Use in a separate reactive cell to update
        the widget when tab selection changes.
        
        Usage:
            # Cell 1: Create widget and tabs
            widget = AnatomagramMultiViewWidget(...)
            tabs = widget.create_view_tabs(mo)
            
            # Cell 2: Create reactive cell to handle tab changes
            @app.cell
            def _(widget, tabs):
                widget.current_view = tabs.value  # Reactive update
                return widget
        
        Args:
            mo: The marimo module for creating UI elements
            
        Returns:
            mo.ui.tabs widget for controlling view switching
        """        
        # Create tab mapping based on available views (value -> label format)
        tab_mapping = {}
        for view in self.available_views:
            if view == "male":
                tab_mapping["male"] = "Male"
            elif view == "female":
                tab_mapping["female"] = "Female"
            elif view == "brain":
                tab_mapping["brain"] = "Brain"
            else:
                tab_mapping[view] = view.title()
        
        # Create tabs widget with initial value from widget's current state
        return mo.ui.tabs(tab_mapping, value=self.current_view)
    
    # Override JavaScript for multi-view functionality
    _esm = """
    import * as d3 from "https://cdn.skypack.dev/d3@7";
    
    // Enhanced responsive scaling helper function (multi-view widget)
    function createResponsiveScaler(containerElement, onResize) {
        let resizeObserver = null;
        let animationFrameId = null;
        
        function measureContainer() {
            const containerRect = containerElement.getBoundingClientRect();
            const computedStyle = getComputedStyle(containerElement);
            const paddingLeft = parseFloat(computedStyle.paddingLeft) || 0;
            const paddingRight = parseFloat(computedStyle.paddingRight) || 0;
            const paddingTop = parseFloat(computedStyle.paddingTop) || 0;
            const paddingBottom = parseFloat(computedStyle.paddingBottom) || 0;
            
            const containerWidth = containerRect.width - paddingLeft - paddingRight;
            const containerHeight = containerRect.height - paddingTop - paddingBottom;
            
            return { containerWidth, containerHeight };
        }
        
        function handleResize() {
            if (animationFrameId) return; // Already scheduled
            
            animationFrameId = requestAnimationFrame(() => {
                try {
                    const { containerWidth, containerHeight } = measureContainer();
                    
                    if (containerWidth > 0 && containerHeight > 0) {
                        onResize(containerWidth, containerHeight);
                    }
                } catch (error) {
                    console.warn('⚠️ Error in resize handler:', error);
                } finally {
                    animationFrameId = null;
                }
            });
        }
        
        // Initialize ResizeObserver (preserved functionality)
        if (window.ResizeObserver) {
            resizeObserver = new ResizeObserver(handleResize);
            resizeObserver.observe(containerElement);
        }
        
        // Return enhanced API with measure and cleanup functions
        return {
            measure: measureContainer,
            cleanup: () => {
                if (resizeObserver) {
                    resizeObserver.disconnect();
                    resizeObserver = null;
                }
                if (animationFrameId) {
                    cancelAnimationFrame(animationFrameId);
                    animationFrameId = null;
                }
            }
        };
    }
    
    export default {
        render({ model, el }) {
            console.log('AnatomagramMultiViewWidget render called');
            console.log('🔍 AnyWidget registered:', !!window.customElements.get('anywidget-view'));
            try { console.log('Element BCR:', el.getBoundingClientRect()); } catch(_) {}
            console.log('Model:', model);
            console.log('Element:', el);
            console.log('Available views:', model.get("available_views"));
            console.log('Current view:', model.get("current_view"));
            
            // Theme detection function - read CSS variables from host document
            function applyThemeStyles() {
                // Try to get the root element (document or marimo container)
                const rootElement = document.documentElement || document.body;
                
                // Get computed styles to read CSS variables from the host document
                const computedStyles = window.getComputedStyle(rootElement);
                
                // Read theme colors from CSS variables defined in external CSS
                const backgroundColor = computedStyles.getPropertyValue('--anatomagram-background').trim() ||
                                     (window.matchMedia('(prefers-color-scheme: dark)').matches ? '#000000' : '#ffffff');
                
                const textColor = computedStyles.getPropertyValue('--anatomagram-text').trim() ||
                                (window.matchMedia('(prefers-color-scheme: dark)').matches ? '#ffffff' : '#333333');
                
                const containerBg = computedStyles.getPropertyValue('--anatomagram-container').trim() ||
                                  (window.matchMedia('(prefers-color-scheme: dark)').matches ? '#101010' : '#f8f8f8');
                
                console.log('🎨 Theme detection:', {
                    backgroundColor,
                    textColor,
                    containerBg,
                    isDarkMode: window.matchMedia('(prefers-color-scheme: dark)').matches
                });
                
                // Apply theme styles directly to container
                mainContainer.style.backgroundColor = backgroundColor;
                mainContainer.style.color = textColor;
                mainContainer.style.borderRadius = '8px';
                
                // Determine if we're in dark mode
                const isDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;

                // Store theme values for other functions to use
                mainContainer._themeColors = {
                    background: backgroundColor,
                    text: textColor,
                    container: containerBg,
                    tooltipBg: computedStyles.getPropertyValue('--anatomagram-tooltip-bg').trim() ||
                              (isDarkMode ? 'rgba(0, 0, 0, 0.95)' : 'rgba(248, 248, 248, 0.95)'),
                    tooltipText: computedStyles.getPropertyValue('--anatomagram-tooltip-text').trim() ||
                               (isDarkMode ? '#ffffff' : '#1b1b1b'),
                    stroke: computedStyles.getPropertyValue('--anatomagram-stroke').trim() ||
                          (isDarkMode ? '#666666' : '#cccccc'),
                    // Semantic colors for tooltip content
                    tooltipSuccess: isDarkMode ? '#66BB6A' : '#4CAF50',
                    tooltipWarning: isDarkMode ? '#FFA726' : '#FF9800',
                    tooltipInfo: isDarkMode ? '#FFD54F' : '#FFC107',
                    tooltipMuted: isDarkMode ? '#999999' : '#666666'
                };
            }
            
            // Create main container
            const mainContainer = document.createElement('div');
            mainContainer.className = 'anatomogram-widget-container';
            // Flexible sizing for marimo compatibility - no min-height, let responsive scaling control size
            mainContainer.style.height = 'auto';
            mainContainer.style.width = '100%';
            mainContainer.style.overflow = 'visible';
            
            // Apply theme before adding to DOM
            applyThemeStyles();
            
            el.appendChild(mainContainer);
            console.log('✅ Container attached with theme. BCR:', mainContainer.getBoundingClientRect());
            
            const container = d3.select(mainContainer)
                .append("div")
                .attr("class", "anatomogram-container");
            
            let currentSvg = null;
            let tooltip = null;
            
            // Multi-view SVG caching for performance optimization
            let multiViewCache = {};
            let multiViewCacheReady = false;
            let cacheInitializationStarted = false;
            
            // Initialize multi-view background SVG caching for all available views
            async function initializeMultiViewSvgCache() {
                if (cacheInitializationStarted) return;
                cacheInitializationStarted = true;
                
                const availableViews = model.get("available_views") || ["male", "female", "brain"];
                console.log('🚀 Starting multi-view SVG cache initialization for views:', availableViews);
                
                try {
                    // Parse all available views asynchronously
                    const parsePromises = availableViews.map(view => parseViewSvg(view));
                    const parseResults = await Promise.all(parsePromises);
                    
                    // Store cached elements by view
                    for (let i = 0; i < availableViews.length; i++) {
                        const view = availableViews[i];
                        multiViewCache[view] = parseResults[i];
                    }
                    
                    multiViewCacheReady = true;
                    console.log('✅ Multi-view SVG cache ready!', {
                        views: Object.keys(multiViewCache),
                        readyStates: Object.keys(multiViewCache).map(v => !!multiViewCache[v])
                    });
                    
                } catch (error) {
                    console.warn('⚠️ Multi-view SVG cache initialization failed:', error);
                    multiViewCacheReady = false;
                }
            }
            
            // Parse individual SVG for a view in background
            function parseViewSvg(view) {
                return new Promise((resolve, reject) => {
                    setTimeout(() => {
                        try {
                            let svgContent;
                            if (view === 'female') {
                                svgContent = model.get("female_svg");
                            } else if (view === 'brain') {
                                svgContent = model.get("brain_svg");
                            } else {
                                svgContent = model.get("male_svg");
                            }
                            
                            if (!svgContent) {
                                console.warn(`No SVG content for view ${view}`);
                                resolve(null);
                                return;
                            }
                            
                            const parser = new DOMParser();
                            const svgDoc = parser.parseFromString(svgContent, "image/svg+xml");
                            
                            if (!svgDoc || !svgDoc.documentElement || svgDoc.documentElement.tagName.toLowerCase() !== 'svg') {
                                throw new Error(`Invalid SVG content for view ${view}`);
                            }
                            
                            console.log(`📦 Cached ${view} view SVG (${svgContent.length} chars)`);
                            resolve(svgDoc.documentElement.cloneNode(true));
                            
                        } catch (error) {
                            console.error(`Error parsing ${view} SVG:`, error);
                            resolve(null);
                        }
                    }, 0); // Yield to event loop
                });
            }
            
            // Initialize tooltip
            function initTooltip() {
                tooltip = d3.select(mainContainer)
                    .append("div")
                    .attr("class", "anatomogram-tooltip")
                    .style("display", "none");
                
                // Apply theme colors to tooltip if available
                if (mainContainer._themeColors) {
                    const theme = mainContainer._themeColors;
                    tooltip
                        .style("background-color", theme.tooltipBg)
                        .style("color", theme.tooltipText)
                        .style("border", `1px solid ${theme.stroke}`)
                        // Set CSS custom properties for semantic colors
                        .style("--tooltip-success-color", theme.tooltipSuccess)
                        .style("--tooltip-warning-color", theme.tooltipWarning)
                        .style("--tooltip-info-color", theme.tooltipInfo)
                        .style("--tooltip-muted-color", theme.tooltipMuted);
                }
            }
            
            // Show loading message
            function showLoading(message = 'Loading anatomagram...') {
                container.html(`<div class="loading-message">${message}</div>`);
            }
            
            // Show error message
            function showError(message) {
                container.html(`<div class="error-message">Error: ${message}</div>`);
            }
            
            // Load SVG based on current_view using multi-view cache or fallback to parsing
            async function loadAnatomagram() {
                const currentView = model.get("current_view");
                
                const hasExistingSvg = !!currentSvg;
                if (!hasExistingSvg) {
                    showLoading(`Loading ${currentView} anatomagram...`);
                }
                
                try {
                    let svgElement;
                    
                    if (hasExistingSvg) {
                        await fadeOutCurrentSvg();
                    }
                    
                    // FAST PATH: Use multi-view cache if available
                    if (multiViewCacheReady && multiViewCache[currentView]) {
                        console.log(`⚡ Loading ${currentView} SVG from multi-view cache (fast path)`);
                        svgElement = multiViewCache[currentView].cloneNode(true);
                        
                    } else {
                        // FALLBACK PATH: Parse on demand if cache not ready
                        console.log(`🐌 Loading ${currentView} SVG with parsing (fallback path)`, {
                            cacheReady: multiViewCacheReady,
                            cacheHasView: !!multiViewCache[currentView],
                            cacheKeys: Object.keys(multiViewCache)
                        });
                        
                        // Get SVG content directly from model traitlets
                        let svgContent;
                        if (currentView === 'female') {
                            svgContent = model.get("female_svg");
                        } else if (currentView === 'brain') {
                            svgContent = model.get("brain_svg");
                        } else {
                            svgContent = model.get("male_svg");
                        }
                        
                        if (!svgContent) {
                            showError("SVG content not available");
                            return;
                        }
                        
                        // Parse SVG content directly
                        const parser = new DOMParser();
                        const svgDoc = parser.parseFromString(svgContent, "image/svg+xml");
                        
                        if (!svgDoc || !svgDoc.documentElement || svgDoc.documentElement.tagName.toLowerCase() !== 'svg') {
                            showError("Invalid SVG content");
                            return;
                        }
                        
                        svgElement = svgDoc.documentElement;
                    }
                    
                    // Common rendering path for both cached and parsed SVGs
                    container.html('');
                    svgElement.style.opacity = '0';
                    container.node().appendChild(svgElement);
                    currentSvg = d3.select(svgElement);
                    
                    // Set SVG size with flexible scaling for marimo
                    const originalViewBox = currentSvg.attr("viewBox");
                    let viewBoxValue = originalViewBox || "0 0 800 1000";
                    
                    // Pre-size SVG using cached dimensions before scaling
                    const sizeScale = model.get("size_scale") || 1.0;
                    const presetDimensions = calculateSvgDimensions(svgElement, cachedContainerWidth, cachedContainerHeight, sizeScale);
                    svgElement.style.width = `${presetDimensions.width}px`;
                    svgElement.style.height = `${presetDimensions.height}px`;
                    
                    // Apply scaling to SVG
                    currentSvg
                        .attr("width", null)
                        .attr("height", null);
                    applySVGScaling();
                    
                    // Set base SVG properties
                    currentSvg
                        .attr("width", null)   // remove width constraint
                        .attr("height", null)  // remove height constraint  
                        .attr("viewBox", viewBoxValue)
                        .attr("preserveAspectRatio", "xMidYMid meet")
                        .style("display", "block") // ensure proper display
                        .style("margin", "0 auto"); // center in container
                    
                    // Attach event handlers and update colors
                    attachEventHandlers();
                    updateColors();
                    
                    fadeInSvgElement(svgElement);
                    
                } catch (error) {
                    console.error("Error loading SVG:", error);
                    showError(`Failed to load anatomogram: ${error.message}`);
                }
            }
            
            // [Note: I'll continue with the rest of the JavaScript functions from the parent class]
            // Since this is getting quite long, I'll include the essential parts and inherit the rest
            
            // Color scale creation (copied from parent)
            function createColorScale(palette, scaleType, minVal, maxVal) {
                const colorSchemes = {
                    'viridis': d3.interpolateViridis,
                    'magma': d3.interpolateMagma,
                    'inferno': d3.interpolateInferno,
                    'plasma': d3.interpolatePlasma,
                    'turbo': d3.interpolateTurbo,
                    'cividis': d3.interpolateCividis,
                    'warm': d3.interpolateWarm,
                    'cool': d3.interpolateCool
                };
                
                const interpolator = colorSchemes[palette] || d3.interpolateViridis;
                
                if (scaleType === 'log') {
                    const logMin = minVal > 0 ? minVal : 0.001;
                    const logScale = d3.scaleLog()
                        .domain([logMin, maxVal])
                        .range([0, 1])
                        .clamp(true);
                    
                    return (value) => {
                        if (value <= 0) return interpolator(0);
                        return interpolator(logScale(value));
                    };
                } else {
                    const linearScale = d3.scaleLinear()
                        .domain([minVal, maxVal])
                        .range([0, 1])
                        .clamp(true);
                    
                    return (value) => interpolator(linearScale(value));
                }
            }
            
            // Helper function to color an element
            function colorElement(element, color, opacity = 1.0) {
                const node = element.node();
                if (node.tagName.toLowerCase() === 'g') {
                    element.selectAll('path, rect, circle, polygon, ellipse')
                        .style('fill', color)
                        .style('opacity', opacity);
                } else {
                    element.style('fill', color)
                        .style('opacity', opacity);
                }
            }
            
            // Update tissue colors based on data (simplified implementation)
            function updateColors() {
                if (!currentSvg) return;

                const selectedItem = model.get("selected_item") || model.get("selected_gene");
                const visualizationData = model.get("visualization_data") || model.get("expression_data");
                const dataType = model.get("data_type") || "expression";
                const palette = model.get("color_palette");
                const scaleType = model.get("scale_type");
                const threshold = model.get("threshold") || 0;
                const noDataOpacity = model.get("no_data_opacity") || 0.5;
                const debugMode = model.get("debug") || false;

                // DEBUG: Log SVG element analysis
                if (debugMode) {
                    const allElements = currentSvg.selectAll('*[id^="UBERON"]');
                    const elementIds = [];
                    allElements.each(function() {
                        elementIds.push(d3.select(this).attr('id'));
                    });

                    console.log('🎨 updateColors DEBUG (MultiViewWidget):', {
                        selectedItem,
                        currentView: model.get("current_view"),
                        totalSvgElements: elementIds.length,
                        svgElementIds: elementIds,
                        dataKeys: visualizationData?.genes?.[selectedItem] ?
                                  Object.keys(visualizationData.genes[selectedItem]) : [],
                        hasData: !!visualizationData?.genes?.[selectedItem]
                    });
                }

                if (!selectedItem || !visualizationData || !visualizationData.genes || !visualizationData.genes[selectedItem]) {
                    if (debugMode) console.warn('⚠️ No valid data for selected item:', selectedItem);
                    // Reset all to default gray if no valid data
                    currentSvg.selectAll('*[id^="UBERON"]').each(function() {
                        const element = d3.select(this);
                        colorElement(element, '#E0E0E0', noDataOpacity);
                    });
                    return;
                }

                const itemData = visualizationData.genes[selectedItem];
                const values = Object.values(itemData).filter(v => typeof v === 'number' && v > 0);

                if (values.length === 0) {
                    if (debugMode) console.warn('⚠️ No valid numeric values in data');
                    currentSvg.selectAll('*[id^="UBERON"]').each(function() {
                        const element = d3.select(this);
                        colorElement(element, '#E0E0E0', noDataOpacity);
                    });
                    return;
                }

                const minVal = d3.min(values) || 0;
                const maxVal = d3.max(values) || 1;

                const colorScale = createColorScale(palette, scaleType, minVal, maxVal);

                // Track coloring stats for debug
                let coloredCount = 0;
                let grayedCount = 0;
                const unmappedElements = [];

                // Update all tissue elements
                currentSvg.selectAll('*[id^="UBERON"]').each(function() {
                    const element = d3.select(this);
                    const tissueId = element.attr('id');
                    const value = itemData[tissueId];

                    if (value !== undefined && value >= threshold) {
                        const color = colorScale(value);
                        colorElement(element, color, 1.0);
                        element.attr('data-expression', value);
                        coloredCount++;
                    } else {
                        colorElement(element, '#E0E0E0', noDataOpacity);
                        element.attr('data-expression', null);
                        grayedCount++;
                        if (debugMode && value === undefined) {
                            unmappedElements.push(tissueId);
                        }
                    }
                });

                // DEBUG: Report coloring results
                if (debugMode) {
                    console.log('🎨 Coloring complete:', {
                        colored: coloredCount,
                        grayed: grayedCount,
                        coveragePercent: (coloredCount / (coloredCount + grayedCount) * 100).toFixed(1) + '%'
                    });

                    if (unmappedElements.length > 0) {
                        console.warn('⚠️ Unmapped SVG elements:', unmappedElements);
                    }
                }
            }

            // Create enhanced tooltip HTML from tooltip content
            function createEnhancedTooltipHTML(content, dataType, uberonDescriptions, tissueId) {
                // Parse multi-line tooltip content
                const lines = content.split('\\n');
                let html = '';

                // Helper function to escape HTML
                function escapeHtml(text) {
                    const div = document.createElement('div');
                    div.textContent = text;
                    return div.innerHTML;
                }

                // Helper function to add description if available
                function addDescription(uberon) {
                    if (uberonDescriptions && uberon && uberonDescriptions[uberon]) {
                        let desc = uberonDescriptions[uberon];
                        // Truncate if longer than 200 characters
                        if (desc.length > 200) {
                            desc = desc.substring(0, 197) + '...';
                        }
                        html += `<div class="tooltip-description">${escapeHtml(desc)}</div>`;
                    }
                }

                // Check if this is a single tissue with mapping info (2 lines) or simple (1 line)
                if (lines.length === 1) {
                    // Simple single-line tooltip: "tissue name: value"
                    const simpleMatch = lines[0].match(/^(.+?):\\s*([\\d\\.\\-eE\\+]+)\\s*$/);
                    if (simpleMatch) {
                        const [_, tissueName, value] = simpleMatch;
                        html += `<div class="tooltip-main-tissue">${escapeHtml(tissueName.trim())}</div>`;
                        const valueLabel = dataType === 'risk' ? 'AD Risk' : 'Expression';
                        html += `<div class="tooltip-main-value">${valueLabel}: ${parseFloat(value).toFixed(3)}</div>`;
                        // Add description for this tissue
                        addDescription(tissueId);
                    } else {
                        // Parsing failed - return null to fall back to basic tooltip
                        console.warn('Single-line enhanced tooltip parsing failed for content:', content);
                        return null;
                    }
                } else if (lines.length === 2) {
                    // Two-line tooltip with mapping: "tissue name: value" + "(mapped from UBERON_...)"
                    const valueMatch = lines[0].match(/^(.+?):\\s*([\\d\\.\\-eE\\+]+)\\s*$/);
                    const mappingMatch = lines[1].match(/^\\(mapped from (.+?)\\)\\s*$/);

                    if (valueMatch && mappingMatch) {
                        const [_, tissueName, value] = valueMatch;
                        const [__, originalUberon] = mappingMatch;
                        html += `<div class="tooltip-main-tissue">${escapeHtml(tissueName.trim())}</div>`;
                        const valueLabel = dataType === 'risk' ? 'AD Risk' : 'Expression';
                        html += `<div class="tooltip-main-value">${valueLabel}: ${parseFloat(value).toFixed(3)}</div>`;
                        // Add description for the original UBERON (before mapping)
                        addDescription(originalUberon);
                        html += `<div class="tooltip-hierarchy-mapping">⚡ Hierarchy mapping from ${escapeHtml(originalUberon)}</div>`;
                    } else {
                        // Parsing failed - return null to fall back to basic tooltip
                        console.warn('Two-line enhanced tooltip parsing failed for content:', content);
                        return null;
                    }
                } else {
                    // Standardized tooltip - region name with contributing tissues
                    const firstLine = lines[0];
                    const standardizedMatch = firstLine.match(/^(.+?) \\((.+?)\\): ([\\d\\.]+)$/);

                    if (standardizedMatch) {
                        const [_, regionName, strategy, value] = standardizedMatch;
                        html += `<div class="tooltip-main-tissue">${escapeHtml(regionName)}</div>`;
                        const valueLabel = dataType === 'risk' ? 'AD Risk' : 'Expression';
                        html += `<div class="tooltip-main-value">${valueLabel}: ${parseFloat(value).toFixed(3)} (${strategy})</div>`;
                        // Add description for this aggregated region
                        addDescription(tissueId);

                        // Find tissues section
                        const fromIndex = lines.findIndex(line => line.startsWith('From ') && line.includes('tissues:'));
                        if (fromIndex > 0) {
                            html += `<div class="tooltip-aggregation-header">Contributing Tissues:</div>`;

                            for (let i = fromIndex + 1; i < lines.length; i++) {
                                const line = lines[i].trim();
                                if (line) {
                                    const tissueMatch = line.match(/^([•◦])\\s*(.+?): ([\\d\\.]+)$/);
                                    if (tissueMatch) {
                                        const [_, indicator, tissueName, tissueValue] = tissueMatch;
                                        const isDirect = indicator === '•';
                                        const indicatorClass = isDirect ? 'direct' : 'hierarchy';
                                        const indicatorSymbol = isDirect ? '●' : '◐';

                                        html += `<div class="tooltip-contributing-tissue">`;
                                        html += `<span><span class="tissue-indicator ${indicatorClass}">${indicatorSymbol}</span>${escapeHtml(tissueName)}</span>`;
                                        html += `<span class="tissue-value">${parseFloat(tissueValue).toFixed(3)}</span>`;
                                        html += `</div>`;
                                    }
                                }
                            }
                        }
                    }
                }

                return html;
            }

            // Attach event handlers with enhanced tooltip support
            function attachEventHandlers() {
                if (!currentSvg || !tooltip) return;

                console.log('🖱️ attachEventHandlers called for MultiViewWidget');
                
                currentSvg.selectAll('*[id^="UBERON"]')
                    .on('mouseover', function(event) {
                        const element = d3.select(this);
                        const tissueId = element.attr('id');
                        const expressionValue = element.attr('data-expression');

                        if (!tissueId) return;

                        const selectedItem = model.get("selected_item") || model.get("selected_gene");
                        const uberonMap = model.get("uberon_map");
                        const enhancedTooltips = model.get("enhanced_tooltips");
                        const uberonDescriptions = model.get("uberon_descriptions");
                        const dataType = model.get("data_type") || "expression";

                        // Use enhanced tooltip if available
                        if (enhancedTooltips && enhancedTooltips[tissueId]) {
                            const enhancedContent = enhancedTooltips[tissueId];

                            // Parse the enhanced content to create rich HTML
                            let tooltipContent = createEnhancedTooltipHTML(enhancedContent, dataType, uberonDescriptions, tissueId);

                            // If enhanced parsing succeeded, use it
                            if (tooltipContent) {
                                tooltip
                                    .style("display", "block")
                                    .html(tooltipContent);
                                return;
                            }
                            // If enhanced parsing failed, fall through to basic tooltip with the enhanced content
                            console.warn('Falling back to basic tooltip for enhanced content:', enhancedContent);
                        }

                        // Fallback to basic tooltip with anatomical region names
                        const uberonNames = model.get("uberon_names");
                        let tissueName;

                        if (uberonMap && uberonMap[tissueId]) {
                            tissueName = uberonMap[tissueId];
                        } else if (uberonNames && uberonNames[tissueId]) {
                            tissueName = uberonNames[tissueId];
                        } else {
                            // Last resort: format the UBERON ID to be more readable
                            tissueName = tissueId.replace('_', ':');
                        }

                        let tooltipContent = `<div class="tooltip-main-tissue">${tissueName}</div>`;

                        if (expressionValue && expressionValue !== 'null') {
                            const value = parseFloat(expressionValue);

                            if (dataType === "risk") {
                                tooltipContent += `<div class="tooltip-main-value">AD Risk: ${value.toFixed(3)}</div>`;
                            } else {
                                tooltipContent += `<div class="tooltip-main-value">Expression: ${value.toFixed(3)}</div>`;
                            }

                            // Add description if available
                            if (uberonDescriptions && tissueId && uberonDescriptions[tissueId]) {
                                let desc = uberonDescriptions[tissueId];
                                // Truncate if longer than 200 characters
                                if (desc.length > 200) {
                                    desc = desc.substring(0, 197) + '...';
                                }
                                // Escape HTML
                                const div = document.createElement('div');
                                div.textContent = desc;
                                tooltipContent += `<div class="tooltip-description">${div.innerHTML}</div>`;
                            }

                            if (selectedItem) {
                                const itemLabel = dataType === "risk" ? "Item" : "Gene";
                                tooltipContent += `<div class="tooltip-item-label">${itemLabel}: ${selectedItem}</div>`;
                            }
                        } else {
                            const noDataLabel = dataType === "risk" ? "No Risk Data Available" : "No Expression Data Available";
                            tooltipContent += `<div class="tooltip-main-value">${noDataLabel}</div>`;
                            tooltipContent += `<div class="tooltip-no-data">This anatomical region has no tissues in the current analysis.</div>`;

                            // Add description even when there's no data
                            if (uberonDescriptions && tissueId && uberonDescriptions[tissueId]) {
                                let desc = uberonDescriptions[tissueId];
                                // Truncate if longer than 200 characters
                                if (desc.length > 200) {
                                    desc = desc.substring(0, 197) + '...';
                                }
                                // Escape HTML
                                const div = document.createElement('div');
                                div.textContent = desc;
                                tooltipContent += `<div class="tooltip-description">${div.innerHTML}</div>`;
                            }
                        }

                        tooltip
                            .style("display", "block")
                            .html(tooltipContent);
                    })
                    .on('mousemove', function(event) {
                        const [x, y] = d3.pointer(event, mainContainer);
                        tooltip
                            .style("left", (x + 10) + "px")
                            .style("top", (y - 10) + "px");
                    })
                    .on('mouseout', function() {
                        tooltip.style("display", "none");
                    });
            }
            
            // Container dimensions for responsive scaling (set by ResizeObserver)
            let containerWidth = 800;
            let containerHeight = 600;
            let cachedContainerWidth = 800;
            let cachedContainerHeight = 600;
            
            function calculateSvgDimensions(svgElement, cachedWidth, cachedHeight, sizeScale) {
                const svgNaturalWidth = svgElement.viewBox?.baseVal?.width || 800;
                const svgNaturalHeight = svgElement.viewBox?.baseVal?.height || 1000;
                
                const widthScale = cachedWidth / svgNaturalWidth;
                const heightScale = cachedHeight / svgNaturalHeight;
                const autoScale = Math.min(widthScale, heightScale);
                
                const minScale = (cachedHeight >= 400) ? (400 / svgNaturalHeight) : 0;
                const finalScale = Math.max(autoScale * sizeScale, minScale);
                
                const maxAllowedHeight = Math.min(svgNaturalHeight * finalScale, 1200);
                const adjustedScale = Math.min(finalScale, maxAllowedHeight / svgNaturalHeight);
                
                const finalWidth = Math.round(svgNaturalWidth * adjustedScale);
                const finalHeight = Math.round(svgNaturalHeight * adjustedScale);
                
                return { width: finalWidth, height: finalHeight };
            }
            
            function applySVGScaling() {
                if (!currentSvg || cachedContainerWidth <= 0 || cachedContainerHeight <= 0) return;
                
                const sizeScale = model.get("size_scale") || 1.0;
                const svgNode = currentSvg.node();
                const dimensions = calculateSvgDimensions(svgNode, cachedContainerWidth, cachedContainerHeight, sizeScale);
                
                const debugMode = model.get("debug") || false;
                if (debugMode) {
                    console.log('📐 MultiView responsive scaling:', {
                        cachedContainerWidth,
                        cachedContainerHeight,
                        sizeScale,
                        finalWidth: dimensions.width,
                        finalHeight: dimensions.height
                    });
                }
                
                currentSvg
                    .style("width", `${dimensions.width}px`)
                    .style("height", `${dimensions.height}px`)
                    .style("max-width", "100%")
                    .style("max-height", "100%");
                
                if (mainContainer) {
                    mainContainer.style.height = `${dimensions.height + 40}px`;
                }
                
                return dimensions;
            }
            
            async function fadeOutCurrentSvg() {
                if (!currentSvg) return;
                const node = currentSvg.node();
                if (!node) {
                    currentSvg = null;
                    return;
                }
                
                // Capture reference and immediately null out currentSvg
                const fadingNode = node;
                currentSvg = null;  // Prevents applySVGScaling() from touching it
                
                // Pin dimensions on the fading node
                const computedStyle = window.getComputedStyle(fadingNode);
                const originalWidth = computedStyle.width;
                const originalHeight = computedStyle.height;
                
                fadingNode.style.width = originalWidth;
                fadingNode.style.height = originalHeight;
                fadingNode.style.maxWidth = originalWidth;
                fadingNode.style.maxHeight = originalHeight;
                
                await new Promise((resolve) => {
                    let resolved = false;
                    const finish = () => {
                        if (resolved) return;
                        resolved = true;
                        fadingNode.removeEventListener('transitionend', handleTransitionEnd);
                        resolve();
                    };
                    const handleTransitionEnd = (event) => {
                        if (event.target === fadingNode) {
                            finish();
                        }
                    };
                    fadingNode.addEventListener('transitionend', handleTransitionEnd);
                    requestAnimationFrame(() => {
                        fadingNode.style.opacity = '0';
                    });
                    setTimeout(finish, 360);
                });
                
                // Use d3 to remove the node
                d3.select(fadingNode).remove();
            }
            
            function fadeInSvgElement(element) {
                if (!element) return;
                requestAnimationFrame(() => {
                    element.style.opacity = '1';
                });
            }
            
            // Initialize
            initTooltip();
            
            // Start multi-view SVG caching immediately for performance
            if (typeof requestIdleCallback !== 'undefined') {
                requestIdleCallback(() => initializeMultiViewSvgCache(), { timeout: 2000 });
            } else {
                setTimeout(() => initializeMultiViewSvgCache(), 100);
            }
            
            // Initialize enhanced ResizeObserver for responsive scaling
            let scaler = null;
            if (mainContainer) {
                scaler = createResponsiveScaler(mainContainer, (width, height) => {
                    if (Math.round(width) !== Math.round(cachedContainerWidth) ||
                        Math.round(height) !== Math.round(cachedContainerHeight)) {
                        cachedContainerWidth = width;
                        cachedContainerHeight = height;
                        containerWidth = width;
                        containerHeight = height;
                        
                        const debugMode = model.get("debug") || false;
                        if (debugMode) {
                            console.log('📏 Container resized - updating cache:', { width, height });
                        }
                        
                        applySVGScaling();
                    }
                });
                
                const previousVisibility = mainContainer.style.visibility;
                mainContainer.style.visibility = 'hidden';
                
                const revealAfterLoad = () => {
                    Promise.resolve(loadAnatomagram()).finally(() => {
                        mainContainer.style.visibility = previousVisibility || '';
                    });
                };
                
                const performInitialRender = (attempt = 0) => {
                    if (!scaler) {
                        revealAfterLoad();
                        return;
                    }
                    
                    const { containerWidth: measuredWidth, containerHeight: measuredHeight } = scaler.measure();
                    
                    if (measuredWidth > 0 && measuredHeight > 0) {
                        cachedContainerWidth = measuredWidth;
                        cachedContainerHeight = measuredHeight;
                        containerWidth = measuredWidth;
                        containerHeight = measuredHeight;
                        applySVGScaling();
                        revealAfterLoad();
                        return;
                    }
                    
                    if (attempt >= 10) {
                        const debugMode = model.get("debug") || false;
                        if (debugMode) {
                            console.warn('⚠️ Container size still zero after retries, rendering with fallback dimensions.');
                        }
                        revealAfterLoad();
                        return;
                    }
                    
                    requestAnimationFrame(() => performInitialRender(attempt + 1));
                };
                
                requestAnimationFrame(() => performInitialRender());
                
                const debugMode = model.get("debug") || false;
                if (debugMode) {
                    console.log('✅ Enhanced ResizeObserver initialized for responsive scaling');
                }
            }
            
            // Listen for property changes (use current_view instead of sex)
            const debugMode = model.get("debug") || false;
            if (debugMode) {
                console.log('🔗 Setting up MultiViewWidget event listeners');
            }
            model.on("change:current_view", () => {
                const debugMode = model.get("debug") || false;
                if (debugMode) {
                    console.log('🔄 current_view changed to:', model.get("current_view"));
                }
                loadAnatomagram();
            });
            model.on("change:selected_gene", () => {
                const debugMode = model.get("debug") || false;
                if (debugMode) {
                    console.log('🧬 selected_gene changed to:', model.get("selected_gene"));
                }
                updateColors();
            });
            model.on("change:selected_item", () => {
                const debugMode = model.get("debug") || false;
                if (debugMode) {
                    console.log('📊 selected_item changed to:', model.get("selected_item"));
                }
                updateColors();
            });
            model.on("change:color_palette", () => {
                const debugMode = model.get("debug") || false;
                if (debugMode) {
                    console.log('🎨 color_palette changed to:', model.get("color_palette"));
                }
                updateColors();
            });
            model.on("change:scale_type", () => {
                const debugMode = model.get("debug") || false;
                if (debugMode) {
                    console.log('📏 scale_type changed to:', model.get("scale_type"));
                }
                updateColors();
            });
            model.on("change:threshold", () => {
                const debugMode = model.get("debug") || false;
                if (debugMode) {
                    console.log('🎯 threshold changed to:', model.get("threshold"));
                }
                updateColors();
            });
            model.on("change:expression_data", () => {
                const debugMode = model.get("debug") || false;
                if (debugMode) {
                    console.log('📈 expression_data changed');
                }
                if (currentSvg) {
                    updateColors();
                } else {
                    loadAnatomagram();
                }
            });
            model.on("change:visualization_data", () => {
                const debugMode = model.get("debug") || false;
                if (debugMode) {
                    console.log('📊 visualization_data changed');
                }
                if (currentSvg) {
                    updateColors();
                } else {
                    loadAnatomagram();
                }
            });
            model.on("change:size_scale", () => {
                const debugMode = model.get("debug") || false;
                if (debugMode) {
                    console.log('📐 size_scale changed to:', model.get("size_scale"));
                }
                applySVGScaling();
            });
            
            // Cleanup on widget destruction
            model.on('comm:close', () => {
                const debugMode = model.get("debug") || false;
                if (debugMode) {
                    console.log('🧹 AnatomagramMultiViewWidget cleanup');
                }
                if (scaler) {
                    scaler.cleanup();
                    scaler = null;
                }
            });
        }
    };
    """
